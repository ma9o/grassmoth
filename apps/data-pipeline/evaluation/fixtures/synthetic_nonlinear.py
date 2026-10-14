"""Synthetic nonlinear SSM fixture for PG/RBPF recovery notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.likelihood import LinkFunction
from nof1_causal_lab.artifacts.model.parameter import ParameterCoordinate, SiteKind
from nof1_causal_lab.artifacts.probability.families import DistributionFamily
from nof1_causal_lab.artifacts.probability.joint_layout import JointLawLayout
from nof1_causal_lab.models.ssm.compile.observation_windows import compile_window_plan
from nof1_causal_lab.models.ssm.preparation.observations import ObservationSupportRuntime
from nof1_causal_lab.models.ssm.preparation.preflight import ObservationPreflightFailure

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs

LATENT_NAMES = [
    "affective_state",
    "sleep_quality",
    "physical_activity",
]

INPUT_NAMES = [
    "serotonergic_exposure",
    "seasonal_load",
]

MANIFEST_NAMES = [
    "state_of_mind_valence",
    "late_night_message_count",
    "total_sleep_hours",
    "sleep_efficiency_pct",
    "sleep_onset_latency_min",
    "nightly_hrv_ms",
    "daily_step_count",
    "exercise_minutes",
    "journal_affect_severity",
    "journal_rumination_intensity",
    "gp_clinical_severity",
]

MANIFEST_DISTS = [
    DistributionFamily.GAUSSIAN,
    DistributionFamily.NEGATIVE_BINOMIAL,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAMMA,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.NEGATIVE_BINOMIAL,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
]

MANIFEST_LINKS = [
    LinkFunction.IDENTITY,
    LinkFunction.LOG,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
    LinkFunction.LOG,
    LinkFunction.IDENTITY,
    LinkFunction.LOG,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
]

TRUE_DECAY = np.asarray([0.28, 0.35, 0.32], dtype=np.float32)
TRUE_HILL_BY_SITE = {
    "vf_3_Emax": 0.06,  # diminishing returns: sleep_quality -> affective_state
    "vf_3_EC50": 0.70,
    "vf_3_n": 2.0,
    "vf_5_Emax": 0.045,  # diminishing returns: physical_activity -> affective_state
    "vf_5_EC50": 0.75,
    "vf_5_n": 2.0,
    "vf_6_Emax": 0.035,  # positive mood improves sleep with saturation
    "vf_6_EC50": 0.80,
    "vf_6_n": 2.0,
    "vf_7_Emax": 0.030,  # behavioral activation saturates
    "vf_7_EC50": 0.80,
    "vf_7_n": 2.0,
}
TRUE_MULTIPLICATIVE_BY_SITE = {
    "vf_4_weight": 0.025,  # sleep and activity reinforce mood together
    "vf_8_weight": 0.020,  # mood and sleep jointly support activity
}
TRUE_OBS_R = 12.0
TRUE_OBS_SHAPE = 8.0

TRUE_DRIFT = np.asarray(
    [
        [-0.28, 0.00, 0.00],
        [0.00, -0.35, 0.00],
        [0.00, 0.00, -0.32],
    ],
    dtype=np.float32,
)

TRUE_DIFFUSION_SD = np.asarray([0.18, 0.16, 0.20], dtype=np.float32)
TRUE_T0_MEAN = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
TRUE_T0_SD = np.asarray([0.55, 0.50, 0.55], dtype=np.float32)

TRUE_INPUT_EFFECT = np.asarray(
    [
        [0.08, 0.03],
        [0.02, 0.00],
        [0.04, 0.00],
    ],
    dtype=np.float32,
)
TRUE_INPUT_EFFECT_POSITIONS = (
    (0, 0),  # serotonergic_exposure -> affective_state
    (0, 1),  # seasonal_load -> affective_state
    (1, 0),  # serotonergic_exposure -> sleep_quality
    (2, 0),  # serotonergic_exposure -> physical_activity
)

TRUE_LOADINGS = np.asarray(
    [
        [0.75, 0.00, 0.00],
        [-0.35, 0.00, 0.00],
        [0.00, 0.35, 0.00],
        [0.00, 2.50, 0.00],
        [0.00, -0.25, 0.00],
        [0.00, 3.00, 0.00],
        [0.00, 0.00, 0.25],
        [0.00, 0.00, 5.00],
        [-0.70, 0.00, 0.00],
        [-1.00, 0.00, 0.00],
        [-1.20, 0.00, 0.00],
    ],
    dtype=np.float32,
)

TRUE_MANIFEST_MEANS = np.asarray(
    [
        0.20,
        np.log(2.0),
        7.10,
        87.0,
        np.log(18.0),
        45.0,
        np.log(80.0),
        28.0,
        2.0,
        4.0,
        5.0,
    ],
    dtype=np.float32,
)

TRUE_MANIFEST_SD = np.asarray(
    [0.25, 1.0, 0.20, 1.00, 1.0, 1.50, 1.0, 2.00, 0.30, 0.40, 0.35],
    dtype=np.float32,
)

_manifest_order = sorted(
    range(len(MANIFEST_NAMES)), key=lambda row: int(np.flatnonzero(TRUE_LOADINGS[row])[0])
)
MANIFEST_NAMES = [MANIFEST_NAMES[i] for i in _manifest_order]
MANIFEST_DISTS = [MANIFEST_DISTS[i] for i in _manifest_order]
MANIFEST_LINKS = [MANIFEST_LINKS[i] for i in _manifest_order]
TRUE_LOADINGS = TRUE_LOADINGS[_manifest_order]
TRUE_MANIFEST_MEANS = TRUE_MANIFEST_MEANS[_manifest_order]
TRUE_MANIFEST_SD = TRUE_MANIFEST_SD[_manifest_order]

EXACT_MEASUREMENT_MANIFEST_INDICES = tuple(
    idx
    for idx, (dist, link) in enumerate(zip(MANIFEST_DISTS, MANIFEST_LINKS, strict=True))
    if (dist == DistributionFamily.GAUSSIAN and link == LinkFunction.IDENTITY)
    or (dist == DistributionFamily.NEGATIVE_BINOMIAL and link == LinkFunction.LOG)
)
EXACT_MEASUREMENT_SUPPORT = np.asarray(
    [idx in EXACT_MEASUREMENT_MANIFEST_INDICES for idx in range(len(MANIFEST_NAMES))],
    dtype=bool,
)
# Gaussian/identity manifest means are PINNED at truth: their raw-scale
# locations (e.g. sleep_efficiency_pct ~ 87) sit tens of prior sd outside the
# canonical Normal(0, 2) manifest-mean prior, so leaving them free makes the
# truth unreachable and distorts every coupled posterior coordinate. Production
# removes the raw location via deterministic centering (bind_panel);
# this fixture bypasses that path, so the location is pinned instead — it is
# not a recovery axis here. Count intercepts (NegBin/log) stay free: their
# log-scale truths sit within the canonical prior.
MEASUREMENT_MEANS_FREE_SUPPORT = np.asarray(
    [
        MANIFEST_DISTS[idx] == DistributionFamily.NEGATIVE_BINOMIAL
        and MANIFEST_LINKS[idx] == LinkFunction.LOG
        for idx in range(len(MANIFEST_NAMES))
    ],
    dtype=bool,
)
ANCHOR_LOADING_POSITIONS = (
    (0, 0),  # state_of_mind_valence anchors affective_state
    (MANIFEST_NAMES.index("total_sleep_hours"), 1),  # sleep anchor
    (MANIFEST_NAMES.index("daily_step_count"), 2),  # activity anchor
)
MEASUREMENT_LOADINGS_FREE_SUPPORT = (
    (~np.isclose(TRUE_LOADINGS, 0.0))
    & EXACT_MEASUREMENT_SUPPORT[:, None]
    & np.asarray(
        [
            [(row, col) not in ANCHOR_LOADING_POSITIONS for col in range(TRUE_LOADINGS.shape[1])]
            for row in range(TRUE_LOADINGS.shape[0])
        ],
        dtype=bool,
    )
)
MEASUREMENT_MEANS_FREE_POSITIONS = tuple(
    idx for idx in range(len(MANIFEST_NAMES)) if bool(MEASUREMENT_MEANS_FREE_SUPPORT[idx])
)
MEASUREMENT_LOADINGS_FREE_POSITIONS = tuple(
    (row, col)
    for row in range(TRUE_LOADINGS.shape[0])
    for col in range(TRUE_LOADINGS.shape[1])
    if bool(MEASUREMENT_LOADINGS_FREE_SUPPORT[row, col])
)


@dataclass(frozen=True)
class SyntheticNonlinearData:
    observations: jnp.ndarray
    times: jnp.ndarray
    latent: jnp.ndarray
    observation_support: ObservationSupportRuntime


def load_synthetic_nonlinear_spec(
    *, T: int = 32, diffusion_scale: float = 1.0
) -> DynamicalModelSpec:
    """Load the retained recovery model and explicitly scale its diffusion priors."""
    dynamical_model_spec = DynamicalModelSpec.model_validate_json(
        Path(__file__).with_name("synthetic_nonlinear_model.json").read_text()
    ).materialized()
    distributions = dict(dynamical_model_spec.distributions)
    for parameter in dynamical_model_spec.parameters:
        if dynamical_model_spec.parameter_context(parameter.id).quantity == SiteKind.DIFFUSION_DIAG:
            assert parameter.distribution is not None
            distributions[parameter.distribution] = dist.HalfNormal(0.4 * float(diffusion_scale))
    inputs = _build_transition_inputs(T)
    input_constructs = tuple(
        sorted(
            (
                construct
                for construct in dynamical_model_spec.constructs
                if construct.role == "exogenous"
            ),
            key=lambda construct: construct.id,
        )
    )
    layout = JointLawLayout(
        parameters=(),
        constructs=tuple(construct.id for construct in input_constructs),
        time_points=tuple(float(index) for index in range(T)),
        labels={},
    )
    distributions[layout.distribution_id] = dist.Delta(
        jnp.asarray(
            np.concatenate(
                [inputs[:, INPUT_NAMES.index(construct.name)] for construct in input_constructs]
            )
        ),
        event_dim=1,
    )
    revised_inputs = tuple(
        construct.with_distribution(layout.distribution_id) for construct in input_constructs
    )
    return dynamical_model_spec.with_entities(
        constructs=tuple(
            construct.with_distribution(layout.distribution_id)
            if construct.role == "exogenous"
            else construct
            for construct in dynamical_model_spec.constructs
        ),
        edges=replace_constructs(dynamical_model_spec.edges, revised_inputs),
        distributions=distributions,
        law_layouts={layout.distribution_id: layout},
    )


def load_synthetic_nonlinear_model(
    *, T: int = 32, diffusion_scale: float = 1.0
) -> CompiledFitInputs:
    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompiledFitInputs,
        compile_ssm_inputs_from_model,
    )
    from nof1_causal_lab.models.structure import StructuralSelection

    inputs = compile_ssm_inputs_from_model(
        StructuralSelection(
            load_synthetic_nonlinear_spec(T=T, diffusion_scale=diffusion_scale), None
        )
    )
    assert isinstance(inputs, CompiledFitInputs), inputs
    return inputs


def bind_synthetic_nonlinear_panel(inputs: CompiledFitInputs, data: SyntheticNonlinearData):
    """Bind generated observations through the same canonical row owner as real data."""
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
    from nof1_causal_lab.artifacts.observations.semantics import AnchorPolicy
    from nof1_causal_lab.models.ssm.preparation.panel import BoundPanel, bind_panel
    from nof1_causal_lab.observations.rows import indicator_window_bounds

    time_origin = datetime(2026, 1, 1, tzinfo=UTC)

    rows = []
    for i, observation in enumerate(inputs.compiled_dynamical_model.observations):
        for j, time in enumerate(np.asarray(data.times)):
            anchor = time_origin + timedelta(days=float(time))
            rows.append(
                {
                    "indicator_id": str(observation.id),
                    "value": None
                    if np.isnan(data.observations[j, i])
                    else float(data.observations[j, i]),
                    "anchor_time": anchor,
                    "support_kind": observation.support.support_kind.value,
                    "summary_operator": observation.support.summary_operator.value,
                    "anchor_policy": observation.support.anchor_policy.value,
                    "observation_window": observation.observation_window.source,
                }
            )
    start, end = indicator_window_bounds(
        pl.col("anchor_time"),
        tuple(item.observation for item in inputs.compiled_dynamical_model.observations),
        anchor_policy=AnchorPolicy.SUPPORT_END,
    )
    observations = ObservationDataset.from_frame(
        pl.DataFrame(rows).with_columns(start.alias("support_start"), end.alias("support_end")),
        tuple(item.observation for item in inputs.compiled_dynamical_model.observations),
        time_origin=time_origin,
    )
    panel = bind_panel(
        observations,
        compiled_dynamical_model=inputs.compiled_dynamical_model,
        time_origin=time_origin,
    )
    assert isinstance(panel, BoundPanel), panel
    return panel


def _build_transition_inputs(T: int) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    serotonergic = 1.0 / (1.0 + np.exp(-(t - 8.0) / 2.0))
    seasonal = np.sin(2.0 * np.pi * t / max(T - 1, 1))
    return np.column_stack([serotonergic, seasonal]).astype(np.float32)


def _build_gp_interval_support(T: int, gp_rows: np.ndarray, window: int = 3):
    n_manifest = len(MANIFEST_NAMES) + len(INPUT_NAMES)
    support_start_times = np.full((T, n_manifest), np.nan, dtype=np.float64)
    support_end_times = np.full((T, n_manifest), np.nan, dtype=np.float64)
    gp_idx = MANIFEST_NAMES.index("gp_clinical_severity")

    for end in gp_rows:
        start = max(0, int(end) - window)
        support_start_times[end, gp_idx] = float(start)
        support_end_times[end, gp_idx] = float(end)

    support_kinds = [None] * n_manifest
    summary_operators = ["last"] * n_manifest
    observation_windows = [None] * n_manifest
    support_kinds[gp_idx] = "interval"
    summary_operators[gp_idx] = "mean"
    observation_windows[gp_idx] = f"{window}d"

    plan = compile_window_plan(
        np.arange(T, dtype=np.float64),
        support_start_times,
        support_end_times,
        support_kinds,
        [*MANIFEST_NAMES, *INPUT_NAMES],
    )
    assert not isinstance(plan, ObservationPreflightFailure), plan
    return ObservationSupportRuntime.assembled(
        window_plan=plan,
        manifest_names=[*MANIFEST_NAMES, *INPUT_NAMES],
        support_kinds=support_kinds,
        summary_operators=summary_operators,
        anchor_policies=("support_end",) * n_manifest,
        observation_windows=observation_windows,
        support_start_times=support_start_times,
        support_end_times=support_end_times,
    )


def _sample_negative_binomial(rng: np.random.Generator, mean: np.ndarray, r: float) -> np.ndarray:
    gamma_rate = rng.gamma(shape=r, scale=mean / r)
    return rng.poisson(gamma_rate).astype(np.float32)


def _hill_effect(source_value: np.ndarray | float, prefix: str) -> np.ndarray | float:
    x = np.maximum(source_value, 0.0)
    emax = TRUE_HILL_BY_SITE[f"{prefix}_Emax"]
    ec50 = TRUE_HILL_BY_SITE[f"{prefix}_EC50"]
    n_hill = TRUE_HILL_BY_SITE[f"{prefix}_n"]
    x_n = x**n_hill
    return emax * x_n / (ec50**n_hill + x_n + 1e-12)


def _synthetic_nonlinear_drift(state: np.ndarray, transition_input: np.ndarray) -> np.ndarray:
    drift = TRUE_DRIFT @ state + TRUE_INPUT_EFFECT @ transition_input
    drift = drift.copy()
    drift[0] += _hill_effect(state[1], "vf_3")
    drift[0] += _hill_effect(state[2], "vf_5")
    drift[1] += _hill_effect(state[0], "vf_6")
    drift[2] += _hill_effect(state[0], "vf_7")
    drift[0] += TRUE_MULTIPLICATIVE_BY_SITE["vf_4_weight"] * state[1] * state[2]
    drift[2] += TRUE_MULTIPLICATIVE_BY_SITE["vf_8_weight"] * state[0] * state[1]
    return drift


def _sample_nonlinear_transition(
    rng: np.random.Generator,
    state: np.ndarray,
    transition_input: np.ndarray,
    *,
    diffusion_sd: np.ndarray,
    dt: float = 1.0,
    substeps: int = 8,
) -> np.ndarray:
    next_state = np.asarray(state, dtype=np.float32).copy()
    step = float(dt) / int(substeps)
    innovation_sd = np.asarray(diffusion_sd, dtype=np.float32) * np.sqrt(step)
    for _ in range(int(substeps)):
        innovation = rng.normal(0.0, innovation_sd).astype(np.float32)
        next_state = next_state + step * _synthetic_nonlinear_drift(next_state, transition_input)
        next_state = next_state + innovation
    return next_state.astype(np.float32)


def simulate_synthetic_nonlinear_data(
    T: int = 32, seed: int = 71, diffusion_scale: float = 1.0
) -> SyntheticNonlinearData:
    rng = np.random.default_rng(seed)
    transition_inputs = _build_transition_inputs(T)
    # Process-noise SD lever: scale=1 is the informative regime; >1 gives diffuse
    # dynamics where the prior-as-proposal advantage of cSMC should erode.
    diffusion_sd = np.asarray(TRUE_DIFFUSION_SD, dtype=np.float32) * float(diffusion_scale)
    latent = np.zeros((T, 3), dtype=np.float32)
    latent[0] = rng.normal(TRUE_T0_MEAN, TRUE_T0_SD).astype(np.float32)
    for time_idx in range(1, T):
        latent[time_idx] = _sample_nonlinear_transition(
            rng,
            latent[time_idx - 1],
            transition_inputs[time_idx],
            diffusion_sd=diffusion_sd,
        )

    linear_predictor = latent @ TRUE_LOADINGS.T + TRUE_MANIFEST_MEANS
    observations = np.full((T, len(MANIFEST_NAMES)), np.nan, dtype=np.float32)
    gaussian_indices = [
        i for i, family in enumerate(MANIFEST_DISTS) if family == DistributionFamily.GAUSSIAN
    ]
    for idx in gaussian_indices:
        observations[:, idx] = rng.normal(
            linear_predictor[:, idx],
            TRUE_MANIFEST_SD[idx],
        ).astype(np.float32)

    observations[:, MANIFEST_NAMES.index("late_night_message_count")] = _sample_negative_binomial(
        rng,
        np.exp(linear_predictor[:, MANIFEST_NAMES.index("late_night_message_count")]),
        TRUE_OBS_R,
    )
    observations[:, MANIFEST_NAMES.index("daily_step_count")] = _sample_negative_binomial(
        rng,
        np.exp(linear_predictor[:, MANIFEST_NAMES.index("daily_step_count")]),
        TRUE_OBS_R,
    )
    gamma_mean = np.exp(linear_predictor[:, MANIFEST_NAMES.index("sleep_onset_latency_min")])
    observations[:, MANIFEST_NAMES.index("sleep_onset_latency_min")] = rng.gamma(
        shape=TRUE_OBS_SHAPE,
        scale=gamma_mean / TRUE_OBS_SHAPE,
    ).astype(np.float32)

    dense_rows = np.arange(T)
    mood_rows = dense_rows[dense_rows % 4 == 0]
    journal_rows = np.asarray([3, 8, 14, 21, 27], dtype=np.int64)
    journal_rows = journal_rows[journal_rows < T]
    gp_rows = np.asarray([6, 18, 30], dtype=np.int64)
    gp_rows = gp_rows[gp_rows < T]

    sparse_keep = {
        0: mood_rows,
        MANIFEST_NAMES.index("journal_affect_severity"): journal_rows,
        MANIFEST_NAMES.index("journal_rumination_intensity"): journal_rows,
        MANIFEST_NAMES.index("gp_clinical_severity"): gp_rows,
    }
    for manifest_idx, rows in sparse_keep.items():
        mask = np.ones(T, dtype=bool)
        mask[rows] = False
        observations[mask, manifest_idx] = np.nan

    support = _build_gp_interval_support(T, gp_rows)
    gp_idx = MANIFEST_NAMES.index("gp_clinical_severity")
    for end in gp_rows:
        start = max(0, int(end) - 3)
        coeff_prev = np.zeros(T, dtype=np.float32)
        coeff_curr = np.zeros(T, dtype=np.float32)
        weights = np.zeros(T, dtype=np.float32)
        coeff_prev[start + 1 : int(end) + 1] = 0.5
        coeff_curr[start + 1 : int(end) + 1] = 0.5
        weights[start + 1 : int(end) + 1] = 1.0
        numerator = np.sum(
            coeff_prev
            * np.concatenate([[linear_predictor[0, gp_idx]], linear_predictor[:-1, gp_idx]])
            + coeff_curr * linear_predictor[:, gp_idx]
        )
        denominator = np.maximum(np.sum(weights), 1e-8)
        observations[end, gp_idx] = rng.normal(
            numerator / denominator,
            TRUE_MANIFEST_SD[gp_idx],
        )

    return SyntheticNonlinearData(
        observations=jnp.asarray(np.column_stack([observations, transition_inputs])),
        times=jnp.arange(T, dtype=jnp.float32),
        latent=jnp.asarray(np.column_stack([latent, transition_inputs])),
        observation_support=support,
    )


def _scalar_recovery_targets() -> dict[str, float]:
    """Bind retained truths through their mechanism identities to current sample sites."""
    from nof1_causal_lab.artifacts.identity import MechanismRef
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.structure.sites import CompiledEdgeTarget, CompiledNodeTarget

    truths = {
        **{
            f"mechanism:relaxation-{name}": float(TRUE_DECAY[index])
            for index, name in enumerate(LATENT_NAMES)
        },
        **{
            f"mechanism:hill-{source}-{target}": TRUE_HILL_BY_SITE[f"{prefix}_Emax"]
            for source, target, prefix in [
                (1, 0, "vf_3"),
                (2, 0, "vf_5"),
                (0, 1, "vf_6"),
                (0, 2, "vf_7"),
            ]
        },
        **{
            f"mechanism:input-{row}-{column}": float(TRUE_INPUT_EFFECT[row, column])
            for row, column in TRUE_INPUT_EFFECT_POSITIONS
        },
        "mechanism:interaction-1-2-0": TRUE_MULTIPLICATIVE_BY_SITE["vf_4_weight"],
        "mechanism:interaction-0-1-2": TRUE_MULTIPLICATIVE_BY_SITE["vf_8_weight"],
    }
    dynamical_model_spec = load_synthetic_nonlinear_spec()
    targets = {"obs_r": TRUE_OBS_R, "obs_shape": TRUE_OBS_SHAPE}
    for binding in parameter_bindings(load_synthetic_nonlinear_model().compiled_dynamical_model)[0]:
        if isinstance(binding.target, (CompiledNodeTarget, CompiledEdgeTarget)):
            mechanism_id = next(
                ref.id
                for ref in dynamical_model_spec.parameter_context(binding.parameter_id).owners
                if isinstance(ref, MechanismRef)
            )
            targets[binding.site.name] = truths[mechanism_id]
    return targets


SCALAR_RECOVERY_TARGETS = _scalar_recovery_targets()


@dataclass(frozen=True)
class RecoveryTargetSpec:
    """A scalar truth and its coordinate in a retained posterior sample site."""

    coordinate: ParameterCoordinate
    true_value: float
    scale: float


MEASUREMENT_MEAN_RECOVERY_TARGETS = {
    f"manifest_mean_{MANIFEST_NAMES[manifest_idx]}": RecoveryTargetSpec(
        coordinate=ParameterCoordinate(site_name="manifest_means_free", indices=(idx,)),
        true_value=float(TRUE_MANIFEST_MEANS[manifest_idx]),
        scale=float(TRUE_MANIFEST_SD[manifest_idx]),
    )
    for idx, manifest_idx in enumerate(MEASUREMENT_MEANS_FREE_POSITIONS)
}

MEASUREMENT_LOADING_RECOVERY_TARGETS = {
    f"loading_{MANIFEST_NAMES[row]}_{LATENT_NAMES[col]}": RecoveryTargetSpec(
        coordinate=ParameterCoordinate(site_name="lambda_free", indices=(idx,)),
        true_value=float(TRUE_LOADINGS[row, col]),
        scale=max(abs(float(TRUE_LOADINGS[row, col])), 1.0),
    )
    for idx, (row, col) in enumerate(MEASUREMENT_LOADINGS_FREE_POSITIONS)
}

RECOVERY_TARGETS: dict[str, RecoveryTargetSpec] = {
    **{
        site: RecoveryTargetSpec(
            coordinate=ParameterCoordinate(site_name=site, indices=()),
            true_value=value,
            scale=max(abs(value), 1.0),
        )
        for site, value in SCALAR_RECOVERY_TARGETS.items()
    },
    **MEASUREMENT_MEAN_RECOVERY_TARGETS,
    **MEASUREMENT_LOADING_RECOVERY_TARGETS,
}
