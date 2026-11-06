"""Tests for config.py: dataclass methods and load_config parsing."""

import textwrap

import pytest
from pydantic import ValidationError

from grassmoth.artifacts.sampling import (
    AdaptedLatentStepSpec,
    AuxiliaryKalmanMethodSpec,
    BlockedPathUpdateSpec,
    ConditionalParameterProposalSpec,
    FixedLatentStepSpec,
    GaussianInitializationSpec,
    GuidedMethodSpec,
    MarginalParticleGibbsSpec,
    NutsDASpec,
    NutsTrajectorySpec,
    PathUpdateSpec,
    SamplerSpec,
)
from grassmoth_server.actions.contracts.settings import FitSettingsSpec
from grassmoth_server.actions.fit.execution import resolve_sampler_spec
from grassmoth_server.config import (
    EmbeddedLLMDefaults,
    ExtractionWorkersConfig,
    InferenceConfig,
    LLMDefaults,
    PipelineConfig,
    get_secret,
    load_config,
    validate_config,
)
from grassmoth_server.llm.specs import EmbeddedLLMSpec

pytestmark = pytest.mark.contract

# =============================================================================
# Owned sampler specification and override resolution
# =============================================================================


class TestSamplerSpec:
    def test_runtime_contract_rejects_unknown_fields(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SamplerSpec.model_validate({"unknown_option": True})

    def test_marginal_particle_gibbs_defaults(self):
        result = InferenceConfig().sampler
        assert isinstance(result.algorithm, MarginalParticleGibbsSpec)
        assert result.algorithm.n_parameter_particles == 2
        assert isinstance(result.algorithm.path_update.method, GuidedMethodSpec)
        assert result.algorithm.path_update.method.kind == "particle_mgrad"
        assert result.algorithm.path_update.method.num_particles == 64
        assert result.algorithm.path_update.method.kappa == 1
        assert result.algorithm.path_update.latent_step == AdaptedLatentStepSpec(target=0.5)
        assert result.algorithm.parameter_proposal == "random_walk"
        encoded = result.model_dump_json()
        assert SamplerSpec.model_validate_json(encoded) == result
        assert result.algorithm.param_target_accept == 0.2
        assert result.algorithm.adaptation_scheme == "dual_averaging"

    def test_public_overrides_resolve_without_mutating_defaults(self, monkeypatch):
        from grassmoth_server import config as config_module

        configured = SamplerSpec(
            num_warmup=10,
            num_samples_per_chain=20,
            num_chains=2,
            seed=3,
            algorithm=MarginalParticleGibbsSpec(
                path_update=PathUpdateSpec(
                    method=GuidedMethodSpec(num_particles=8),
                    latent_step=FixedLatentStepSpec(delta=0.31),
                )
            ),
        )
        config = _make_pipeline_config()
        monkeypatch.setattr(
            config_module,
            "get_config",
            lambda: PipelineConfig(
                extraction_workers=config.extraction_workers,
                inference=InferenceConfig(sampler=configured),
            ),
        )
        resolved = resolve_sampler_spec(
            FitSettingsSpec(
                num_warmup=0,
                num_samples_per_chain=30,
                num_chains=4,
                seed=0,
                time_limit_seconds=90,
            )
        )
        assert (
            resolved.num_warmup,
            resolved.num_samples_per_chain,
            resolved.num_chains,
            resolved.seed,
            resolved.time_limit_seconds,
        ) == (0, 30, 4, 0, 90)
        assert isinstance(resolved.algorithm, MarginalParticleGibbsSpec)
        assert isinstance(configured.algorithm, MarginalParticleGibbsSpec)
        assert resolved.algorithm.path_update is configured.algorithm.path_update
        assert resolved.algorithm.path_update.latent_step == FixedLatentStepSpec(delta=0.31)
        assert (
            configured.num_warmup,
            configured.num_samples_per_chain,
            configured.num_chains,
            configured.seed,
        ) == (10, 20, 2, 3)
        assert resolve_sampler_spec(FitSettingsSpec()) == configured
        assert resolve_sampler_spec(
            FitSettingsSpec(algorithm=MarginalParticleGibbsSpec(adaptation_scheme="simple"))
        ) == (configured.revised(algorithm=MarginalParticleGibbsSpec(adaptation_scheme="simple")))
        assert configured.time_limit_seconds == 14400
        assert resolve_sampler_spec(FitSettingsSpec(), resolved).time_limit_seconds == 90
        assert (
            resolve_sampler_spec(
                FitSettingsSpec(time_limit_seconds=30), resolved
            ).time_limit_seconds
            == 30
        )
        kalman = resolve_sampler_spec(
            FitSettingsSpec(
                algorithm=MarginalParticleGibbsSpec(
                    path_update=PathUpdateSpec(
                        method=AuxiliaryKalmanMethodSpec(),
                        update=BlockedPathUpdateSpec(block_length=300),
                        proposal=ConditionalParameterProposalSpec(),
                    )
                )
            )
        )
        assert kalman == configured.revised(
            algorithm=MarginalParticleGibbsSpec(
                path_update=PathUpdateSpec(
                    method=AuxiliaryKalmanMethodSpec(),
                    update=BlockedPathUpdateSpec(block_length=300),
                    proposal=ConditionalParameterProposalSpec(),
                )
            )
        )
        nuts = resolve_sampler_spec(
            FitSettingsSpec.model_validate(
                {
                    "algorithm": {
                        "method": "nuts_da",
                        "parameterization": "mixed",
                        "trajectory": {"kind": "nuts", "max_tree_depth": 5},
                    }
                }
            )
        )
        assert nuts.algorithm == NutsDASpec(
            parameterization="mixed", trajectory=NutsTrajectorySpec(max_tree_depth=5)
        )
        assert resolve_sampler_spec(FitSettingsSpec(), nuts).algorithm is nuts.algorithm
        reset = resolve_sampler_spec(
            FitSettingsSpec.model_validate({"algorithm": {"method": "mpgibbs"}}), nuts
        )
        assert reset.algorithm == MarginalParticleGibbsSpec()
        with pytest.raises(ValidationError, match="path_update"):
            FitSettingsSpec.model_validate(
                {
                    "path_update": {
                        "method": {"kind": "auxiliary_kalman"},
                        "update": {"kind": "blocked", "block_length": 300},
                        "proposal": {"kind": "conditional_parameter"},
                    }
                }
            )
        with pytest.raises(ValidationError, match="frozen"):
            resolved.seed = 8  # ty: ignore[invalid-assignment] -- Exercise runtime rejection of a frozen field.

    def test_custom_chains_and_seed(self):
        result = InferenceConfig(sampler=SamplerSpec(num_chains=8, seed=42)).sampler
        assert result.num_chains == 8
        assert result.seed == 42


# =============================================================================
# load_config (with temp config file)
# =============================================================================


MINIMAL_CONFIG = textwrap.dedent("""\
    extraction_workers:
      llm:
        model: openrouter/gpt-4
""")

FULL_CONFIG = textwrap.dedent("""\
    llm:
      embedded:
        max_tokens: 4096
        timeout: 120
        reasoning_effort: low

    extraction_workers:
      max_concurrent_workers: 6
      max_tool_turns: 45
      llm:
        model: openrouter/claude-3

    inference:
      compute_loo_diagnostics: false
      sampler:
        num_warmup: 500
        num_samples_per_chain: 2000
        num_chains: 2
        seed: 123
        algorithm:
          method: mpgibbs
          n_parameter_particles: 3
          path_update:
            method:
              kind: twisted_particle_agrad
              num_particles: 24
            update: {kind: whole_path}
            proposal: {kind: parameter_mean}
            latent_step:
              kind: fixed
              delta: 0.29
          param_target_accept: 0.4
          initialization:
            scale: 0.25
""")


class TestLoadConfig:
    @pytest.mark.parametrize(
        "invalid",
        [
            "llm:\n  embedded:\n    reasoning_effort: invalid\n",
            "inference:\n  hardware: imaginary_gpu\n",
            "inference:\n  sampler:\n    algorithm:\n      method: nonexistent\n",
            "inference:\n  sampler:\n    n_parameter_particles: 1\n",
            "inference:\n  sampler:\n    marginal_particle_gibbs:\n      init_method: invalid\n",
            "inference:\n  sampler:\n    num_chains: many\n",
            "inference:\n  unknown_setting: true\n",
        ],
    )
    def test_rejects_invalid_yaml_before_returning_typed_config(self, tmp_path, invalid):
        path = tmp_path / "config.yaml"
        path.write_text(MINIMAL_CONFIG + invalid)
        load_config.cache_clear()
        with pytest.raises(ValidationError):
            load_config(path)

    def test_load_minimal(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(MINIMAL_CONFIG)

        load_config.cache_clear()

        import grassmoth_server.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.extraction_workers.max_concurrent_workers == 4
        assert cfg.extraction_workers.max_tool_turns == 40
        # Defaults for optional sections
        assert cfg.llm.embedded.max_tokens == 65536

        load_config.cache_clear()

    def test_load_full(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import grassmoth_server.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.extraction_workers.max_concurrent_workers == 6
        assert cfg.extraction_workers.max_tool_turns == 45
        assert cfg.inference.sampler.num_warmup == 500
        assert cfg.inference.sampler.num_samples_per_chain == 2000
        assert cfg.inference.sampler.num_chains == 2
        assert cfg.inference.sampler.seed == 123
        assert cfg.inference.compute_loo_diagnostics is False
        assert isinstance(cfg.inference.sampler.algorithm, MarginalParticleGibbsSpec)
        assert isinstance(cfg.inference.sampler.algorithm.path_update.method, GuidedMethodSpec)
        assert cfg.inference.sampler.algorithm.path_update.method.num_particles == 24
        assert cfg.inference.sampler.algorithm.n_parameter_particles == 3
        assert cfg.inference.sampler.algorithm.path_update.latent_step == FixedLatentStepSpec(
            delta=0.29
        )
        assert cfg.inference.sampler.algorithm.path_update.method.kind == "twisted_particle_agrad"
        assert cfg.inference.sampler.algorithm.param_target_accept == 0.4
        assert cfg.inference.sampler.algorithm.initialization == GaussianInitializationSpec(
            scale=0.25
        )
        assert cfg.llm.embedded.max_tokens == 4096
        assert cfg.llm.embedded.reasoning_effort == "low"

        load_config.cache_clear()

    def test_sampler_config_roundtrip(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import grassmoth_server.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        sampler = cfg.inference.sampler
        assert isinstance(sampler.algorithm, MarginalParticleGibbsSpec)
        assert sampler.num_warmup == 500
        assert sampler.algorithm.initialization == GaussianInitializationSpec(scale=0.25)

        load_config.cache_clear()


# =============================================================================
# validate_config
# =============================================================================


def _make_pipeline_config(**profile_llm_overrides) -> PipelineConfig:
    """Build a valid PipelineConfig with optional per-context llm overrides."""
    defaults = {
        "extraction_workers": EmbeddedLLMSpec(model="openrouter/x"),
    }
    defaults.update(profile_llm_overrides)
    return PipelineConfig(
        extraction_workers=ExtractionWorkersConfig(llm=defaults["extraction_workers"]),
        inference=InferenceConfig(),
        llm=LLMDefaults(embedded=EmbeddedLLMDefaults()),
    )


class TestValidateConfig:
    def test_happy_path_all_embedded(self):
        config = _make_pipeline_config()
        assert validate_config(config) == []

    def test_embedded_model_must_be_openrouter_prefix(self):
        config = _make_pipeline_config(
            extraction_workers=EmbeddedLLMSpec(model="gpt-5.4"),
        )
        assert any("openrouter/" in error for error in validate_config(config))


# =============================================================================
# get_secret
# =============================================================================


class TestGetSecret:
    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_ABC", "from-env")
        assert get_secret("TEST_SECRET_ABC") == "from-env"

    def test_returns_none_when_missing(self, monkeypatch):
        monkeypatch.delenv("DEFINITELY_NOT_SET_XYZ_789", raising=False)
        assert get_secret("DEFINITELY_NOT_SET_XYZ_789") is None
