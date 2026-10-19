"""Tests for config.py: dataclass methods and load_config parsing."""

import textwrap

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.actions.fit.execution import resolve_sampler_spec
from nof1_causal_lab.actions.temporal.llm.messages import LLMSubroutineInput, LLMSubroutineRef
from nof1_causal_lab.artifacts.sampling import (
    AdaptedLatentStepSpec,
    DsmcSpec,
    FitSettingsSpec,
    FixedLatentStepSpec,
    InferenceHardware,
    LapsInitializationSpec,
    MarginalParticleGibbsSpec,
    NutsDASpec,
    ParticleGibbsSpec,
    SamplerSpec,
)
from nof1_causal_lab.config import (
    ClaudeCodeDefaults,
    CodexDefaults,
    EmbeddedLLMDefaults,
    ExtractionWorkersConfig,
    InferenceConfig,
    LLMDefaults,
    PiDefaults,
    PipelineConfig,
    get_secret,
    load_config,
    validate_config,
)
from nof1_causal_lab.llm.specs import CodexLLMSpec, EmbeddedLLMSpec, LLMProfileSpec

pytestmark = pytest.mark.contract

# =============================================================================
# Owned sampler specification and override resolution
# =============================================================================


class TestSamplerSpec:
    @pytest.mark.parametrize(
        "algorithm",
        [
            {"method": "custom"},
            {"method": "csmc"},
            {"method": "pgibbs", "ancestor_sampling": True},
            {"method": "pgibbs", "dsmc": {"ancestor_sampling": True}},
            {"method": "pgibbs", "dsmc": {"dsmc_leaf_proposal": "amala_exact"}},
            {"method": "mpgibbs", "dsmc": {"dsmc_leaf_proposal": "amala_exact"}},
        ],
    )
    def test_sampler_choices_reject_removed_or_unsupported_algorithms(self, algorithm):
        with pytest.raises(ValidationError):
            SamplerSpec.model_validate({"algorithm": algorithm})

    def test_runtime_contract_rejects_unknown_fields(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SamplerSpec.model_validate({"unknown_option": True})

    def test_marginal_particle_gibbs_defaults(self):
        result = InferenceConfig().sampler
        assert result.num_particles == 64
        options = result.algorithm
        assert isinstance(options, MarginalParticleGibbsSpec)
        assert options.n_parameter_particles == 2
        assert options.dsmc.drift == "none"
        assert options.dsmc.pilot_mixture is True
        assert options.dsmc.latent_block_coords == 1
        assert options.dsmc.latent_step == AdaptedLatentStepSpec(target=0.5)
        assert options.parameter_proposal == "random_walk"
        encoded = result.model_dump_json()
        assert SamplerSpec.model_validate_json(encoded) == result
        assert options.param_step_size == 0.02
        assert options.param_target_accept == 0.35
        assert result.retain_latent_paths is True

    def test_public_overrides_resolve_without_mutating_defaults(self, monkeypatch):
        from nof1_causal_lab import config as config_module

        configured = SamplerSpec(
            num_warmup=10,
            num_samples_per_chain=20,
            num_chains=2,
            seed=3,
            num_particles=8,
            algorithm=MarginalParticleGibbsSpec(
                dsmc=DsmcSpec(latent_step=FixedLatentStepSpec(delta=0.31))
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
                num_particles=16,
                time_limit_seconds=90,
            )
        )
        assert (
            resolved.num_warmup,
            resolved.num_samples_per_chain,
            resolved.num_chains,
            resolved.seed,
            resolved.num_particles,
            resolved.time_limit_seconds,
        ) == (0, 30, 4, 0, 16, 90)
        assert resolved.algorithm is configured.algorithm
        assert isinstance(resolved.algorithm, MarginalParticleGibbsSpec)
        assert resolved.algorithm.dsmc.latent_step == FixedLatentStepSpec(delta=0.31)
        assert (
            configured.num_warmup,
            configured.num_samples_per_chain,
            configured.num_chains,
            configured.seed,
            configured.num_particles,
        ) == (10, 20, 2, 3, 8)
        assert resolve_sampler_spec(FitSettingsSpec()) == configured
        assert configured.time_limit_seconds == 14400
        assert resolve_sampler_spec(FitSettingsSpec(), resolved).time_limit_seconds == 90
        assert (
            resolve_sampler_spec(
                FitSettingsSpec(time_limit_seconds=30), resolved
            ).time_limit_seconds
            == 30
        )
        assert (
            resolve_sampler_spec(FitSettingsSpec(algorithm=NutsDASpec())).algorithm == NutsDASpec()
        )
        with pytest.raises(ValidationError, match="frozen"):
            resolved.seed = 8  # ty: ignore[invalid-assignment] -- Exercise runtime rejection of a frozen field.

    @pytest.mark.parametrize("hardware", tuple(InferenceHardware))
    @pytest.mark.parametrize(
        "algorithm", [NutsDASpec(), MarginalParticleGibbsSpec(), ParticleGibbsSpec()]
    )
    def test_hardware_and_algorithm_are_independent(self, hardware, algorithm):
        config = TypeAdapter(InferenceConfig).validate_python(
            {
                "sampler": {"hardware": hardware.value, "algorithm": algorithm.model_dump()},
            }
        )
        assert config.sampler.hardware is hardware
        assert config.sampler.algorithm == algorithm
        assert InferenceConfig().sampler.hardware is InferenceHardware.L40S

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
        harness: none
        model: openrouter/gpt-4
""")

FULL_CONFIG = textwrap.dedent("""\
    llm:
      embedded:
        max_tokens: 4096
        timeout: 120
        reasoning_effort: low
      claude_code:
        effort: medium
      codex:
        reasoning_effort: medium
        service_tier: fast
      pi:
        bin: pi-custom
        provider: openai-codex
        thinking: high

    extraction_workers:
      max_concurrent_workers: 6
      max_tool_turns: 45
      llm:
        harness: none
        model: openrouter/claude-3

    inference:
      compute_loo_diagnostics: false
      sampler:
        num_warmup: 500
        num_samples_per_chain: 2000
        num_chains: 2
        seed: 123
        num_particles: 24
        algorithm:
          method: mpgibbs
          n_parameter_particles: 3
          dsmc:
            latent_step:
              kind: fixed
              delta: 0.29
            drift: amala
          param_step_size: 0.03
          param_step_size_min: 0.000001
          param_step_size_max: 0.7
          param_target_accept: 0.4
          adaptation_rate: 0.03
          initialization:
            kind: laps
            ensemble_size: 256
            num_unadjusted_steps: 1000
            num_adjusted_steps: 20
""")


class TestLoadConfig:
    @pytest.mark.parametrize(
        "invalid",
        [
            "llm:\n  embedded:\n    reasoning_effort: invalid\n",
            "llm:\n  codex:\n    reasoning_effort: invalid\n",
            "inference:\n  hardware: imaginary_gpu\n",
            "inference:\n  sampler:\n    algorithm:\n      method: map\n",
            "inference:\n  sampler:\n    algorithm:\n      method: nuts_da\n      target_accept_prob: 1.0\n",
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

        import nof1_causal_lab.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.extraction_workers.max_concurrent_workers == 4
        assert cfg.extraction_workers.max_tool_turns == 40
        # Defaults for optional sections
        assert cfg.llm.embedded.max_tokens == 65536
        assert cfg.llm.codex.reasoning_effort == "xhigh"
        assert cfg.llm.codex.service_tier == "fast"
        assert cfg.llm.pi.provider == "openai-codex"
        assert cfg.llm.pi.thinking == "high"

        load_config.cache_clear()

    def test_load_full(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import nof1_causal_lab.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.extraction_workers.max_concurrent_workers == 6
        assert cfg.extraction_workers.max_tool_turns == 45
        assert cfg.inference.sampler.num_warmup == 500
        assert cfg.inference.sampler.num_samples_per_chain == 2000
        assert cfg.inference.sampler.num_chains == 2
        assert cfg.inference.sampler.seed == 123
        assert cfg.inference.compute_loo_diagnostics is False
        assert cfg.inference.sampler.num_particles == 24
        assert isinstance(cfg.inference.sampler.algorithm, MarginalParticleGibbsSpec)
        assert cfg.inference.sampler.algorithm.n_parameter_particles == 3
        assert cfg.inference.sampler.algorithm.dsmc.latent_step == FixedLatentStepSpec(delta=0.29)
        assert cfg.inference.sampler.algorithm.dsmc.drift == "amala"
        assert cfg.inference.sampler.algorithm.param_step_size == 0.03
        assert cfg.inference.sampler.algorithm.param_step_size_min == 1e-6
        assert cfg.inference.sampler.algorithm.param_step_size_max == 0.7
        assert cfg.inference.sampler.algorithm.param_target_accept == 0.4
        assert cfg.inference.sampler.algorithm.adaptation_rate == 0.03
        assert isinstance(cfg.inference.sampler.algorithm.initialization, LapsInitializationSpec)
        assert cfg.inference.sampler.algorithm.initialization.ensemble_size == 256
        assert cfg.inference.sampler.algorithm.initialization.num_unadjusted_steps == 1000
        assert cfg.inference.sampler.algorithm.initialization.num_adjusted_steps == 20
        assert cfg.llm.embedded.max_tokens == 4096
        assert cfg.llm.embedded.reasoning_effort == "low"
        assert cfg.llm.claude_code.effort == "medium"
        assert cfg.llm.codex.reasoning_effort == "medium"
        assert cfg.llm.codex.service_tier == "fast"
        assert cfg.llm.pi == PiDefaults(bin="pi-custom", provider="openai-codex", thinking="high")

        load_config.cache_clear()

    def test_sampler_config_roundtrip(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import nof1_causal_lab.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        sampler = cfg.inference.sampler
        assert sampler.num_warmup == 500
        assert isinstance(sampler.algorithm, MarginalParticleGibbsSpec)
        assert isinstance(sampler.algorithm.initialization, LapsInitializationSpec)
        assert sampler.algorithm.initialization.ensemble_size == 256

        load_config.cache_clear()


# =============================================================================
# validate_config
# =============================================================================


def _make_pipeline_config(**profile_llm_overrides) -> PipelineConfig:
    """Build a valid PipelineConfig with optional per-context llm overrides."""
    defaults = {
        "extraction_workers": EmbeddedLLMSpec(harness="none", model="openrouter/x"),
    }
    defaults.update(profile_llm_overrides)
    return PipelineConfig(
        extraction_workers=ExtractionWorkersConfig(llm=defaults["extraction_workers"]),
        inference=InferenceConfig(),
        llm=LLMDefaults(
            embedded=EmbeddedLLMDefaults(),
            claude_code=ClaudeCodeDefaults(),
            codex=CodexDefaults(),
        ),
    )


class TestValidateConfig:
    def test_happy_path_all_embedded(self):
        config = _make_pipeline_config()
        assert validate_config(config) == []

    @pytest.mark.parametrize("harness", ["claude-code", "codex", "pi"])
    def test_extraction_workers_require_embedded_backend(self, harness):
        with pytest.raises(ValidationError, match=r"llm\.harness"):
            TypeAdapter(ExtractionWorkersConfig).validate_python(
                {"llm": {"harness": harness, "model": "worker-model"}}
            )

    def test_embedded_model_must_be_openrouter_prefix(self):
        config = _make_pipeline_config(
            extraction_workers=EmbeddedLLMSpec(model="gpt-5.4"),
        )
        assert any("openrouter/" in error for error in validate_config(config))

    @pytest.mark.parametrize(
        "payload",
        [
            {"harness": "anthropic"},
            {"harness": "none", "effort": "high"},
            {"harness": "none", "service_tier": "fast"},
            {"harness": "none", "reasoning_effort": "max"},
            {"harness": "claude-code", "reasoning_effort": "high"},
            {"harness": "claude-code", "effort": "ultra"},
            {"harness": "codex", "max_budget_usd": 5.0},
            {"harness": "codex", "reasoning_effort": "minimal"},
            {"harness": "pi", "reasoning_effort": "high"},
            {"harness": "pi", "thinking": "max"},
        ],
    )
    def test_profile_schema_rejects_incompatible_backend_fields(self, payload):
        with pytest.raises(ValidationError):
            TypeAdapter(LLMProfileSpec).validate_python({"model": "openrouter/x", **payload})

    def test_codex_max_effort_survives_default_resolution_and_transport(self):
        llm = CodexLLMSpec(model="codex-test", bin="custom-codex", reasoning_effort="max")
        request = LLMSubroutineInput(
            subroutine=LLMSubroutineRef(
                workspace_id="test",
                run_id="run",
                subroutine_id="raw_data",
                context_ref="context.json",
            ),
            llm=llm,
            max_tool_turns=5,
        )
        restored = LLMSubroutineInput.model_validate_json(request.model_dump_json())
        assert restored.subroutine == request.subroutine
        assert restored.llm == CodexLLMSpec(
            model="codex-test",
            bin="custom-codex",
            reasoning_effort="max",
        )

    def test_load_config_raises_on_stage2_harness_violation(self, tmp_path, monkeypatch):
        bad_config = textwrap.dedent("""\
            extraction_workers:
              llm:
                harness: claude-code
                model: sonnet
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(bad_config)

        load_config.cache_clear()

        import nof1_causal_lab.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        with pytest.raises(ValueError, match=r"extraction_workers\.llm\.harness"):
            load_config()

        load_config.cache_clear()


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


# =============================================================================
# ensure_harness_prereqs
# =============================================================================


class TestEnsureHarnessPrereqs:
    def _reset(self):
        from nof1_causal_lab import config as config_module

        config_module._verified_harnesses.clear()

    def test_missing_openrouter_key_raises_for_embedded(self, monkeypatch):
        from nof1_causal_lab.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            ensure_harness_prereqs("none")

    def test_caches_successful_check(self, monkeypatch):
        """Once verified, removing the env var doesn't re-trigger the check."""
        from nof1_causal_lab.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ensure_harness_prereqs("none")

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Still cached — does not raise.
        ensure_harness_prereqs("none")
