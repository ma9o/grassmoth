"""Tiny synthetic values for rendering stories, never scientific execution.

These declarations choose illustrative UI states. Public value constructors own
all parsing; no preparation, inference, diagnostics or simulation runs here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from grassmoth.public import (
    DataDiffOutput,
    DynamicalModelSpec,
    EditModelOutput,
    FitOutput,
    NumericalArray,
    PrepareDataOutput,
    SimulateOutput,
)

BASELINE = "construct:00000000000000000001"
TREATMENT = "construct:00000000000000000002"
OUTCOME = "construct:00000000000000000003"
TREATMENT_READING = "indicator:00000000000000000001"
OUTCOME_READING = "indicator:00000000000000000002"
DECAY = "parameter:" + "1".zfill(64)
DIFFUSION = "parameter:" + "2".zfill(64)
ORIGIN = "2026-01-01T00:00:00Z"
TIMES = (0, 1, 2, 3)


def oid(seq: int) -> str:
    """Stable example identity; these are explicitly synthetic, not Git commits."""
    return f"{seq:07x}".ljust(40, "0")


def array(values: object) -> NumericalArray:
    """Encode explicit example coordinates with the production numerical owner."""
    return NumericalArray.from_numpy(np.asarray(values))


def observation(identity: str, name: str, *, resolved: bool = False) -> dict[str, object]:
    """One continuous reading with an optional resolved daily support window."""
    return {
        "id": identity,
        "name": name,
        "measurement_dtype": "continuous",
        "aggregation": "mean",
        "observation_window": "1d" if resolved else None,
    }


OBSERVATIONS = (
    observation(TREATMENT_READING, "treatment_reading", resolved=True),
    observation(OUTCOME_READING, "outcome_reading", resolved=True),
)
EXTRACTION = {
    "kind": "semantic",
    "how_to_measure": "Synthetic example reading.",
    "source_columns": [],
}


def model(stage: Literal["structure", "measurement", "authored", "edited"]) -> EditModelOutput:
    """Progressively populated definitions, parsed by the public model contract."""
    measured = stage != "structure"
    authored = stage in {"authored", "edited"}
    constructs: dict[str, object] = {}
    for identity, name in ((BASELINE, "baseline"), (TREATMENT, "treatment"), (OUTCOME, "outcome")):
        indicators = {}
        if measured and identity != BASELINE:
            reading = TREATMENT_READING if identity == TREATMENT else OUTCOME_READING
            indicators[reading] = {
                "observation": {
                    key: value
                    for key, value in observation(reading, f"{name}_reading").items()
                    if key != "id"
                },
                "construct_polarity": "positive",
                "likelihood": {
                    "law": {
                        "distribution": "Normal",
                        "loc": {"kind": "state", "construct_id": identity},
                        "scale": {
                            "kind": "coefficient",
                            "role": "observation_scale",
                            "value": 1,
                            "construct_ids": [],
                        },
                    },
                    "standardized": False,
                    "reasoning": "Synthetic reading law for interface coverage.",
                }
                if authored
                else None,
            }
        constructs[identity] = {
            "name": name,
            "description": f"Synthetic {name} for interface examples.",
            "role": "endogenous",
            "temporal_status": "time_invariant" if identity == BASELINE else "time_varying",
            "indicators": indicators,
            "dynamics": {
                (
                    "mechanism:00000000000000000001"
                    if identity == OUTCOME
                    else "mechanism:00000000000000000002"
                ): {
                    "kind": "drift",
                    "expression": {
                        "kind": "binary",
                        "operator": "multiply",
                        "left": {
                            "kind": "coefficient",
                            "role": "decay",
                            "value": DECAY if identity == OUTCOME else 0.5,
                            "construct_ids": [],
                        },
                        "right": {
                            "kind": "binary",
                            "operator": "subtract",
                            "left": {"kind": "literal", "value": 0},
                            "right": {"kind": "state", "construct_id": identity},
                        },
                    },
                },
            }
            if authored and identity != BASELINE
            else {},
            "coefficients": (
                [
                    {
                        "kind": "coefficient",
                        "role": "initial_mean",
                        "value": 0,
                        "construct_ids": [],
                    },
                    {
                        "kind": "coefficient",
                        "role": "initial_scale",
                        "value": 1,
                        "construct_ids": [],
                    },
                ]
                + (
                    [
                        {
                            "kind": "coefficient",
                            "role": "diffusion_scale",
                            "value": DIFFUSION if identity == OUTCOME else 0.2,
                            "construct_ids": [],
                        }
                    ]
                    if identity != BASELINE
                    else []
                )
            )
            if authored
            else [],
        }
    spec = DynamicalModelSpec.model_validate(
        {
            "constructs": constructs,
            "edges": {
                "edge:00000000000000000001": {
                    "cause": BASELINE,
                    "effect": TREATMENT,
                    "description": "Baseline affects treatment.",
                    "mechanisms": {
                        "mechanism:00000000000000000003": {
                            "kind": "drift",
                            "expression": {
                                "kind": "binary",
                                "operator": "multiply",
                                "left": {
                                    "kind": "coefficient",
                                    "role": "weight",
                                    "value": 0.2,
                                    "construct_ids": [],
                                },
                                "right": {"kind": "state", "construct_id": BASELINE},
                            },
                        }
                    }
                    if authored
                    else {},
                },
                "edge:00000000000000000002": {
                    "cause": TREATMENT,
                    "effect": OUTCOME,
                    "mechanisms": {
                        "mechanism:00000000000000000004": {
                            "kind": "drift",
                            "expression": {
                                "kind": "binary",
                                "operator": "multiply",
                                "left": {
                                    "kind": "coefficient",
                                    "role": "weight",
                                    "value": 0.5 if stage == "edited" else 0.3,
                                    "construct_ids": [],
                                },
                                "right": {"kind": "state", "construct_id": TREATMENT},
                            },
                        }
                    }
                    if authored
                    else {},
                    "description": "Revised synthetic assumption."
                    if stage == "edited"
                    else "Treatment affects the outcome.",
                },
            },
            "parameters": {
                DECAY: {
                    "name": "outcome_persistence",
                    "description": "Synthetic persistence.",
                    "transform": {"kind": "dt_persistence_to_ct_decay", "interval_days": 1},
                    "distribution": "distribution:00000000000000000001",
                },
                DIFFUSION: {
                    "name": "outcome_diffusion",
                    "description": "Synthetic noise.",
                    "transform": {"kind": "identity"},
                    "distribution": "distribution:00000000000000000002",
                },
            }
            if authored
            else {},
            "distributions": {
                "distribution:00000000000000000001": {
                    "distribution": "Beta",
                    "params": {"concentration1": 2, "concentration0": 2},
                },
                "distribution:00000000000000000002": {
                    "distribution": "HalfNormal",
                    "params": {"scale": 1},
                },
            }
            if authored
            else {},
            "measurement_clock": "1d" if authored else None,
        }
    ).materialized()
    return EditModelOutput.model_validate(
        {
            "dynamical_model_spec": spec,
            "checks": {
                "specification": []
                if authored
                else [
                    {
                        "kind": "not_evaluated",
                        "code": "specification",
                        "subject": "specification",
                        "reason": "MODEL_INCOMPLETE",
                        "detail": "This synthetic model is still being authored.",
                    }
                ],
                "question": {"findings": []},
            },
            "identification": {
                "outcome": OUTCOME,
                "treatments": {
                    TREATMENT: {
                        "status": "identified",
                        "estimand": "Synthetic identified-effect display example.",
                    }
                },
            },
            "pruning": {"constructs": [], "edges": [], "parameters": [], "distributions": []},
        }
    )


def history(values: tuple[int, ...]) -> dict[str, object]:
    """Aligned four-point example, with explicit daily support intervals."""
    return {
        "times": TIMES,
        "values": values,
        "support_start": (-1, 0, 1, 2),
        "support_end": TIMES,
        "empirical": [],
    }


def preparation() -> PrepareDataOutput:
    """A prepared panel without any extraction or data-quality execution."""
    return PrepareDataOutput.model_validate(
        {
            "data": {
                TREATMENT_READING: history((0, 1, 1, 1)),
                OUTCOME_READING: history((1, 3, 2, 5)),
            },
            "metadata": {
                "preparation": {
                    "default_window": "1d",
                    "variables": [
                        {"observation": item, "extraction": EXTRACTION} for item in OBSERVATIONS
                    ],
                    "context": "Synthetic story panel.",
                },
                "time_origin": ORIGIN,
                "variables": OBSERVATIONS,
            },
            "profile": {"indicators": {}, "findings": []},
            "extraction": {"workers": [], "extraction_reused": None},
        }
    )


def fit() -> FitOutput:
    """Four authored joint atoms and their display summaries; no sampler is run."""
    authored = model("authored").dynamical_model_spec
    # This literal is the identity of the declared layout. The model owner verifies it.
    identity = "distribution:6d0f6bb90f381a89d242859072863a6eb50c03e9f36630399f805f28fc742144"
    atoms = array([[0.1, 0.2], [0.3, 0.4], [0.7, 0.6], [0.9, 0.8]])
    parameters = [
        (DECAY, "outcome_persistence", (0.1, 0.3, 0.7, 0.9)),
        (DIFFUSION, "outcome_diffusion", (0.2, 0.4, 0.6, 0.8)),
    ]
    layout = {
        "parameters": [(key, [key.replace("parameter:", "element:")]) for key, _, _ in parameters],
        "constructs": [],
        "time_points": [],
        "time_origin": "relative",
        "labels": {key.replace("parameter:", "element:"): name for key, name, _ in parameters},
    }
    conditioned = authored.revised(
        parameter_definitions={
            parameter.id: parameter.revised(
                distribution=identity, transform={"kind": "identity"}
            ).model_dump(mode="json", exclude={"id"})
            for parameter in authored.parameters
        },
        distribution_definitions={
            identity: {
                "distribution": "MixtureSameFamily",
                "params": {
                    "mixing_distribution": {
                        "distribution": "CategoricalProbs",
                        "params": {"probs": [0.25, 0.25, 0.25, 0.25]},
                    },
                    "component_distribution": {
                        "distribution": "Delta",
                        "params": {"v": atoms, "event_dim": 1},
                    },
                },
            }
        },
        layout_definitions={identity: layout},
    ).materialized()
    density = {"x": [0, 0.5, 1], "density": [0, 2, 0]}
    return FitOutput.model_validate(
        {
            "dynamical_model_spec": conditioned,
            "checks": {
                "data": {"indicators": {}, "findings": []},
                "preflight": [],
                "question": {"findings": []},
            },
            "inference": {
                "evidence": {},
                "core": {
                    "inference_metadata": {
                        "distribution": identity,
                        "num_samples_total": 4,
                        "num_chains": 2,
                        "duration_seconds": 0,
                        "engine": {"method": "mpgibbs"},
                        "sampler_diagnostics": None,
                    },
                    "inference_diagnostics": None,
                    "convergence": {
                        "findings": [],
                        "checked": 0,
                        "status": "not_evaluated",
                        "messages": ["Synthetic display example; no sampler was run."],
                        "warnings": [],
                    },
                    "posterior_marginals": [
                        {
                            "parameter": name,
                            "subject": {
                                "parameter_id": key,
                                "element_id": key.replace("parameter:", "element:"),
                            },
                            "density_curve": density,
                            "empirical": [
                                {"value": value, "probability": probability}
                                for value, probability in zip(
                                    values, (0.25, 0.5, 0.75, 1), strict=True
                                )
                            ],
                            "mean": 0.5,
                            "lower": values[0],
                            "upper": values[-1],
                            "interval_kind": "hdi",
                            "interval_mass": 0.9,
                            "sd": 0.3,
                        }
                        for key, name, values in parameters
                    ],
                    "prior_densities": {DECAY: density, DIFFUSION: density},
                },
                "detail": {
                    "trace_data": [
                        {
                            "subject": {
                                "parameter_id": key,
                                "element_id": key.replace("parameter:", "element:"),
                            },
                            "chains": [
                                {
                                    "array": atoms,
                                    "indices": [None, column],
                                    "start": start,
                                    "stop": start + 2,
                                }
                                for start in (0, 2)
                            ],
                        }
                        for column, (key, _, _) in enumerate(parameters)
                    ]
                },
            },
        }
    )


def simulation(*, causal: bool) -> SimulateOutput:
    """Two explicitly authored paired paths; these values were never simulated."""
    reference = array(
        [[[0, 0, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4]], [[0, 0, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]]]
    )
    action = array(
        [[[0, 0, 1], [0, 2, 3], [0, 2, 4], [0, 2, 5]], [[0, 0, 2], [0, 2, 5], [0, 2, 6], [0, 2, 7]]]
    )
    observed = array([[[0, 1], [2, 3], [2, 4], [2, 5]], [[0, 2], [2, 5], [2, 6], [2, 7]]])
    return SimulateOutput.model_validate(
        {
            "data": [
                {TREATMENT_READING: history((0, 2, 2, 2)), OUTCOME_READING: history(values)}
                for values in ((1, 3, 4, 5), (2, 5, 6, 7))
            ],
            "report": {
                "evidence": {
                    "assignments": [{"target": TREATMENT, "time": 1, "value": 2}],
                    "time_origin": ORIGIN,
                    "times": TIMES,
                    "draws": 2,
                    "seed": 0,
                    "state_ids": [BASELINE, TREATMENT, OUTCOME],
                    "parameter_draws": {},
                    "arms": {
                        "kind": "paired",
                        "action": {"latent_paths": action, "observations": observed},
                        "reference": {
                            "latent_paths": reference,
                            "observations": array(
                                [[[0, 1], [1, 2], [1, 3], [1, 4]], [[0, 2], [1, 3], [1, 4], [1, 5]]]
                            ),
                        },
                        "causal": {
                            "outcome": OUTCOME,
                            "labels": {OUTCOME: "outcome"},
                            "differences": array([[0, 1, 1, 1], [0, 2, 2, 2]]),
                            "frame": [0, 2],
                            "summary": {
                                "mean": 1.5,
                                "median": 1.5,
                                "lower_95": 1,
                                "upper_95": 2,
                                "prob_positive": 1,
                            },
                            "reference_mean": 4.5,
                            "manifest_effects": {OUTCOME_READING: 1.5},
                            "warnings": ["Synthetic interface example."],
                        }
                        if causal
                        else {
                            "kind": "not_evaluated",
                            "code": "causal_effect",
                            "subject": "causal_effect",
                            "reason": "CAUSAL_EVALUATION_FAILED",
                            "detail": "Synthetic example of an unavailable causal effect.",
                        },
                    },
                    "observation_layout": {
                        "variables": OBSERVATIONS,
                        "support_start_times": array([[-1, -1], [0, 0], [1, 1], [2, 2]]),
                        "support_end_times": array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                        "mask": array(np.ones((2, 4, 2), dtype=bool)),
                    },
                },
                "summary": {
                    "state_frames": {BASELINE: [-1, 1], TREATMENT: [0, 2], OUTCOME: [1, 7]},
                    "indicator_frames": {TREATMENT_READING: [0, 2], OUTCOME_READING: [1, 7]},
                    "action_category_probabilities": {},
                    "reference_category_probabilities": {},
                },
                "law": {"kind": "authored"},
                "fit_reliability": "not_fitted",
                "findings": [],
            },
        }
    )


def comparison() -> DataDiffOutput:
    """Declared warning and aligned plot evidence for comparison UI coverage."""
    return DataDiffOutput.model_validate(
        {
            "report": {
                "left": [{"revision": oid(105), "replicate_index": 0}],
                "right": [{"revision": oid(9), "replicate_index": index} for index in (0, 1)],
                "variables": [
                    {
                        "indicator_id": OUTCOME_READING,
                        "changes": [],
                        "findings": [],
                        "statistics": [{"statistic": "mean", "left": [2.75], "right": [3.25, 5]}],
                        "predictive": {
                            "reference_side": "left",
                            "evaluation": {
                                "n_subsample": 2,
                                "findings": [
                                    {
                                        "kind": "evaluated",
                                        "code": "calibration",
                                        "subject": {
                                            "target": {"kind": "indicator", "id": OUTCOME_READING}
                                        },
                                        "outcome": "failed",
                                        "evidence": {
                                            "criterion": "calibration",
                                            "note": "Synthetic calibration warning for interface coverage.",
                                            "value": 0.5,
                                            "lower": 0.7,
                                            "upper": 0.98,
                                        },
                                    }
                                ],
                                "overlays": [
                                    {
                                        "indicator_id": OUTCOME_READING,
                                        "times": TIMES,
                                        "time_origin": ORIGIN,
                                        "standardized": False,
                                        "observed": [1, 3, 2, 5],
                                        "median": [1.5, 4, 5, 6],
                                        "spaghetti_draws": [[1, 3, 4, 5], [2, 5, 6, 7]],
                                    }
                                ],
                                "test_stats": [
                                    {
                                        "indicator_id": OUTCOME_READING,
                                        "stat_name": "mean",
                                        "observed_value": 2.75,
                                        "rep_values": [3.25, 5],
                                        "p_value": 0,
                                        "histogram": [
                                            {
                                                "bin_center": 3.25,
                                                "bin_start": 3,
                                                "bin_end": 4,
                                                "count": 1,
                                            },
                                            {
                                                "bin_center": 5,
                                                "bin_start": 4,
                                                "bin_end": 6,
                                                "count": 1,
                                            },
                                        ],
                                    }
                                ],
                            },
                        },
                    }
                ],
            }
        }
    )
