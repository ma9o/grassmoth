"""Support code for the sampler-pieces walkthrough notebook.

The notebook runs the production sampler, through the study's public actions over
HTTP, on one small model, and checks every result against that model's exact
posterior.

The model is one latent level x on a grid of 45-minute steps over eight days,
pulled back towards zero (decay 0.5 per day), pushed by five known on/off inputs
through unknown effects β, shaken by an unknown noise scale σ, and read with noise
at every grid point. It is linear and Gaussian, so a Kalman filter gives the exact
posterior. It keeps the two features that decide our sampler choice for France:

- the effects are *loose*: the readings pin them about as well with or without the
  exact path;
- the noise scale is *coupled*: given a whole path, σ is pinned to about 1/√(2T) on
  the log scale, far tighter than the readings alone pin it.

Everything here is a client of the actions (`edit_question`, `edit_model`,
`simulate`, `fit`) plus exact Kalman arithmetic for the answer key; no sampler code
is re-implemented.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type Json = dict[str, Any]
type Chart = Literal["centred", "non_centred"]
type PathUpdate = Literal["twisted_particle_agrad", "particle_rwm"]

# ── The study client ────────────────────────────────────────────────────────────────

SERVER = "http://localhost:8100"
WORKSPACE = "PIECES_20261026"


class ActionFailed(RuntimeError):
    """An action finished with a failure; its messages say why."""


@dataclass(frozen=True)
class Study:
    """Submit an action, wait for it, and read the artifacts it produced.

    Identical inputs return the saved call, so re-running the notebook replays every
    result without recomputing it.
    """

    server: str = SERVER
    workspace: str = WORKSPACE

    def _request(self, url: str, payload: Json | None = None) -> Json:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        request = urllib.request.Request(
            url, data=data, headers=headers, method="GET" if data is None else "POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise ActionFailed(
                f"{url}: HTTP {error.code}: {error.read().decode()[:2000]}"
            ) from error

    def run(self, action: str, payload: Json) -> Json:
        """Submit one action and poll until it finishes; return its envelope."""
        base = f"{self.server}/api/studies/{self.workspace}/{action}"
        envelope = self._request(base, {"action": action, "input": payload})
        while envelope["status"] not in ("success", "failed"):
            time.sleep(2)
            envelope = self._request(f"{base}/{envelope['call_id']}")
        if envelope["status"] == "failed":
            errors = [m.get("detail") for m in envelope["messages"] if m.get("severity") == "error"]
            raise ActionFailed(f"{action} failed: {errors}")
        return envelope

    def produced(self, commit_id: str) -> dict[str, str]:
        """Artifact revisions a call produced, by artifact name."""
        timeline = self._request(f"{self.server}/api/studies/{self.workspace}/timeline")
        for attempt in timeline["attempts"]:
            if attempt.get("commit_id") == commit_id:
                effects = attempt["record"]["attempt"]["outcome"]["effects"]
                return {item["artifact_id"]: item["revision"] for item in effects["produced"]}
        raise ActionFailed(f"no journal entry {commit_id}")

    def question(self) -> str:
        payload = {"question": {"text": QUESTION, "outcome": "construct:x", "queries": {}}}
        return self.produced(self.run("edit_question", payload)["commit_id"])["question"]

    def model(self, question: str, spec: Json) -> str:
        envelope = self.run("edit_model", {"parent_ref": question, "dynamical_model_spec": spec})
        return self.produced(envelope["commit_id"])["model"]

    def simulate(self, model: str, *, seed: int = 7) -> tuple[str, Json]:
        """One simulated history; returns the data revision and the body."""
        payload = {
            "dynamical_model_spec_ref": model,
            "simulation": {"start": "2020-01-01", "horizon": f"{DAYS}d", "interventions": []},
            "seed": seed,
            "num_replicates": 1,
        }
        envelope = self.run("simulate", payload)
        return envelope["commit_id"], envelope["body"]

    def fit(self, model: str, data: str, algorithm: Json) -> Json:
        payload = {
            "target": {
                "kind": "new",
                "dynamical_model_spec_ref": model,
                "data_ref": {"revision": data, "replicate_index": 0},
            },
            "settings": {**SCHEDULE, "algorithm": algorithm},
        }
        return self.run("fit", payload)["body"]


# ── The model, in the study's modelling language ────────────────────────────────────

QUESTION = "Sampler pieces: how do known inputs move a latent level?"
EFFECTS = 5
TRUE_EFFECTS = (1.0, -0.6, 0.5, 0.8, -0.4)
TRUE_SCALE = 0.5
DECAY = 0.5
INITIAL_SCALE = 0.1
SCALE_PRIOR = (math.log(0.5), 1.0)
DAYS = 8
STEPS = 256
CLOCK = "45m"
NOISY = 0.3

TIME_POINTS = tuple(i * DAYS / STEPS for i in range(STEPS + 1))


def input_wave(index: int, day: float) -> float:
    """On/off input with its own period, so each effect leaves its own trace."""
    return 1.0 if math.sin(2 * math.pi * (index + 1) * day / DAYS + 0.7 * index) > 0 else 0.0


def _parameter_id(name: str) -> str:
    return "parameter:" + hashlib.sha256(name.encode()).hexdigest()


def _law_id(construct: str) -> str:
    """Joint laws are named by the hash of their coordinates (`JointLawLayout.distribution_id`)."""
    payload = [[], [[construct, list(TIME_POINTS)]], "relative"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "distribution:" + hashlib.sha256(encoded.encode()).hexdigest()


def _state(construct: str) -> Json:
    return {"kind": "state", "construct_id": construct}


def _coefficient(role: str, value: float | str) -> Json:
    return {"kind": "coefficient", "role": role, "value": value, "construct_ids": []}


def _binary(operator: str, left: Json, right: Json) -> Json:
    return {"kind": "binary", "operator": operator, "left": left, "right": right}


def _restoring(construct: str) -> Json:
    """−decay·(x − centre) − quartic·(x − centre)³ with quartic 0: linear mean reversion."""
    offset = _binary("subtract", _state(construct), _coefficient("center", 0.0))
    pull = _binary(
        "multiply",
        _binary("multiply", {"kind": "literal", "value": -1.0}, _coefficient("decay", DECAY)),
        offset,
    )
    cubic = _binary(
        "multiply",
        _coefficient("quartic", 0.0),
        _binary("power", offset, {"kind": "literal", "value": 3.0}),
    )
    return _binary("subtract", pull, cubic)


def _indicator(name: str, law: Json) -> Json:
    return {
        "observation": {
            "name": name,
            "measurement": {"dtype": "continuous"},
            "aggregation": "last",
            "observation_window": CLOCK,
        },
        "likelihood": {"law": law, "standardized": False, "reasoning": "Notebook model"},
        "construct_polarity": "positive",
    }


def model_spec(*, reading_sd: float, free: bool) -> Json:
    """The notebook's model: the effects and σ as priors (free) or at their true values."""
    scale: float | str = _parameter_id("sigma") if free else TRUE_SCALE
    constructs: Json = {
        "construct:x": {
            "name": "x",
            "description": "Latent level pushed by the inputs",
            "role": "endogenous",
            "temporal_status": "time_varying",
            "coefficients": [
                _coefficient("diffusion_scale", scale),
                _coefficient("initial_mean", 0.0),
                _coefficient("initial_scale", INITIAL_SCALE),
            ],
            "dynamics": {
                "mechanism:self-x": {"kind": "drift", "expression": _restoring("construct:x")}
            },
            "indicators": {
                "indicator:x": _indicator(
                    "x",
                    {
                        "distribution": "Normal",
                        "loc": _state("construct:x"),
                        "scale": _coefficient("observation_scale", reading_sd),
                    },
                )
            },
        }
    }
    edges: Json = {}
    parameters: Json = {}
    distributions: Json = {}
    layouts: Json = {}
    for j in range(EFFECTS):
        construct = f"construct:u{j + 1}"
        law = _law_id(construct)
        constructs[construct] = {
            "name": f"u{j + 1}",
            "description": f"Known on/off input {j + 1}",
            "role": "exogenous",
            "temporal_status": "time_varying",
            "distribution": law,
            "indicators": {
                f"indicator:u{j + 1}": _indicator(
                    f"u{j + 1}", {"distribution": "Delta", "v": _state(construct)}
                )
            },
        }
        distributions[law] = {
            "distribution": "Delta",
            "params": {
                "v": {"array": [input_wave(j, day) for day in TIME_POINTS], "dtype": "float32"},
                "event_dim": 1,
            },
        }
        layouts[law] = {
            "parameters": [],
            "constructs": [construct],
            "time_points": list(TIME_POINTS),
            "time_origin": "relative",
            "labels": {},
        }
        weight: float | str = _parameter_id(f"beta{j + 1}") if free else TRUE_EFFECTS[j]
        edges[f"edge:u{j + 1}-x"] = {
            "cause": construct,
            "effect": "construct:x",
            "description": f"Input {j + 1} pushes x",
            "mechanisms": {
                f"mechanism:u{j + 1}-x": {
                    "kind": "drift",
                    "expression": _binary(
                        "multiply", _coefficient("weight", weight), _state(construct)
                    ),
                }
            },
        }
        if free:
            parameters[_parameter_id(f"beta{j + 1}")] = {
                "name": f"beta{j + 1}",
                "description": f"Effect of input {j + 1}",
                "distribution": f"distribution:prior-beta{j + 1}",
            }
            distributions[f"distribution:prior-beta{j + 1}"] = {
                "distribution": "Normal",
                "params": {"loc": 0.0, "scale": 1.0},
            }
    if free:
        parameters[_parameter_id("sigma")] = {
            "name": "sigma",
            "description": "Noise scale of x",
            "distribution": "distribution:prior-sigma",
        }
        distributions["distribution:prior-sigma"] = {
            "distribution": "LogNormal",
            "params": {"loc": SCALE_PRIOR[0], "scale": SCALE_PRIOR[1]},
        }
    return {
        "constructs": constructs,
        "edges": edges,
        "parameters": parameters,
        "distributions": distributions,
        "law_layouts": layouts,
        "measurement_clock": CLOCK,
    }


# ── Sampler settings, copied from the PARAM policy runs ─────────────────────────────

SCHEDULE: Json = {
    "num_warmup": 1000,
    "num_samples_per_chain": 1000,
    "num_chains": 4,
    "seed": 1,
    "hardware": "local",
    "stop_when_converged": None,
}


def mpgibbs(*, path_update: PathUpdate = "twisted_particle_agrad") -> Json:
    """The MPGibbs sampler with a chosen path update."""
    update: Json = (
        {
            "method": path_update,
            "num_particles": 128,
            "latent_step": {"kind": "adapted", "target": 0.5},
            "kappa": 1,
        }
        if path_update == "twisted_particle_agrad"
        else {"method": path_update, "num_particles": 128}
    )
    return {
        "method": "mpgibbs",
        "n_parameter_particles": 2,
        "tempering": {"kind": "untempered"},
        "path_update": update,
        "parameter_proposal": "random_walk",
        "diagnostic_metrics": ["parameter_movement", "particle_identity", "frequency_refresh"],
        "param_target_accept": 0.35,
        "adaptation_scheme": "simple",
        "initialization": {"scale": 0.1},
    }


def joint_nuts(chart: Chart) -> Json:
    """Joint NUTS over the parameters and every path coordinate, in one chart."""
    return {"method": "nuts_da", "parameterization": chart}


# ── Reading the actions' outputs ────────────────────────────────────────────────────


@dataclass(frozen=True)
class History:
    """One history as the fit sees it: grid times (days), readings (NaN where unread), inputs."""

    times: np.ndarray
    readings: np.ndarray
    inputs: np.ndarray
    reading_sd: float

    @property
    def read(self) -> np.ndarray:
        return ~np.isnan(self.readings)

    def refined(self, factor: int) -> History:
        """The same readings on a grid with ``factor`` steps between consecutive points."""
        if factor == 1:
            return self
        fractions = np.arange(factor) / factor
        gaps = np.diff(self.times)
        times = np.append(
            (self.times[:-1, None] + gaps[:, None] * fractions[None]).ravel(), self.times[-1]
        )
        readings = np.full(times.size, np.nan)
        readings[::factor] = self.readings
        inputs = np.concatenate(
            [np.repeat(self.inputs[:, :-1], factor, axis=1), self.inputs[:, -1:]], axis=1
        )
        return History(times, readings, inputs, self.reading_sd)


def history(body: Json, *, reading_sd: float) -> History:
    """The simulated readings of x, with the inputs the model owns."""
    channel = body["data"][0]["indicator:x"]
    values = np.asarray(channel["values"]["array"], dtype=float)
    index = tuple(slice(None) if i is None else i for i in channel["values"]["indices"])
    times = np.asarray(channel["times"], dtype=float)
    inputs = np.array([[input_wave(j, day) for day in times] for j in range(EFFECTS)])
    return History(times, values[index], inputs, reading_sd)


@dataclass(frozen=True)
class FitResult:
    """Retained draws, per-sweep telemetry and native diagnostics of one fit."""

    names: tuple[str, ...]
    draws: np.ndarray
    paths: np.ndarray
    extra: Mapping[str, np.ndarray]
    ess: Mapping[str, float]
    r_hat: Mapping[str, float]
    path_ess: float
    verdict: str

    def column(self, name: str) -> np.ndarray:
        """(chains, draws) of one parameter, with σ on the log scale."""
        values = self.draws[..., self.names.index(name)]
        return np.log(values) if name == "sigma" else values


def read_fit(body: Json) -> FitResult:
    """Unpack a fit body: the conditioned model holds the retained draws, chain by chain."""
    spec = body["dynamical_model_spec"]
    inference = body["inference"]
    core = inference["core"]
    law = core["inference_metadata"]["distribution"]
    layout = spec["law_layouts"][law]
    names = tuple(spec["parameters"][parameter]["name"] for parameter, _ in layout["parameters"])
    table = np.asarray(
        spec["distributions"][law]["params"]["component_distribution"]["params"]["v"], dtype=float
    )
    chains = core["inference_metadata"]["num_chains"]
    table = table.reshape(chains, table.shape[0] // chains, table.shape[1])
    diagnostics = core["inference_diagnostics"]
    latent = diagnostics.get("per_latent") or [{"ess_bulk": float("nan")}]
    extra = {
        key: np.asarray(value)
        for key, value in (inference["evidence"]["chain_extra_fields"] or {}).items()
    }
    return FitResult(
        names=names,
        draws=table[..., : len(names)],
        paths=table[..., len(names) :],
        extra=extra,
        ess={p["parameter"]: float(p["ess_bulk"]) for p in diagnostics.get("per_parameter") or []},
        r_hat={p["parameter"]: float(p["r_hat"]) for p in diagnostics.get("per_parameter") or []},
        path_ess=float(latent[0]["ess_bulk"]),
        verdict=core["convergence"]["status"],
    )


# ── The exact answer (Kalman filter, NumPy float64 on the CPU) ──────────────────────


def _step(history: History, k: int) -> tuple[float, np.ndarray]:
    """Euler–Maruyama step k (from point k−1): its length and the inputs held over it."""
    return float(history.times[k] - history.times[k - 1]), history.inputs[:, k - 1]


def _kalman(history: History, log_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log p(readings | σ) with the path and the effects integrated out, per σ on a grid."""
    count = history.inputs.shape[0]
    mean = np.zeros((log_scales.size, count + 1))
    cov = np.zeros((log_scales.size, count + 1, count + 1))
    cov[:, 0, 0] = INITIAL_SCALE**2
    cov[:, 1:, 1:] = np.eye(count)
    variance = np.exp(2 * log_scales)
    log_lik = np.zeros(log_scales.size)
    for k in range(history.times.size):
        if k:
            dt, held = _step(history, k)
            step = np.eye(count + 1)
            step[0, 0] = 1 - DECAY * dt
            step[0, 1:] = held * dt
            mean = mean @ step.T
            cov = np.einsum("ij,gjk,lk->gil", step, cov, step)
            cov[:, 0, 0] += variance * dt
        if history.read[k]:
            total = cov[:, 0, 0] + history.reading_sd**2
            gain = cov[:, :, 0] / total[:, None]
            innovation = history.readings[k] - mean[:, 0]
            log_lik += -0.5 * (np.log(2 * np.pi * total) + innovation**2 / total)
            mean = mean + gain * innovation[:, None]
            cov = cov - gain[:, :, None] * gain[:, None, :] * total[:, None, None]
    return log_lik, mean[:, 1:], cov[:, 1:, 1:]


def _log_scale_prior(log_scales: np.ndarray) -> np.ndarray:
    return -0.5 * ((log_scales - SCALE_PRIOR[0]) / SCALE_PRIOR[1]) ** 2


@dataclass(frozen=True)
class Exact:
    """The exact posterior of θ = (β1..β5, log σ), tabulated on a grid of log σ.

    Given σ the effects' posterior is Gaussian, so ``effect_mean`` and ``effect_cov``
    complete the joint posterior.
    """

    log_scales: np.ndarray
    density: np.ndarray
    effect_mean: np.ndarray
    effect_cov: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        return self.density * (self.log_scales[1] - self.log_scales[0])

    @property
    def scale_mean(self) -> float:
        return float(self.weights @ self.log_scales)

    @property
    def scale_sd(self) -> float:
        return float(np.sqrt(self.weights @ (self.log_scales - self.scale_mean) ** 2))

    @property
    def effect_means(self) -> np.ndarray:
        return self.weights @ self.effect_mean

    @property
    def effect_sd(self) -> np.ndarray:
        second = np.einsum("g,gk->k", self.weights, np.diagonal(self.effect_cov, axis1=1, axis2=2))
        return np.sqrt(second + self.weights @ self.effect_mean**2 - self.effect_means**2)

    def effect_density(self, index: int, values: np.ndarray) -> np.ndarray:
        """Exact marginal density of one effect: a mixture over the σ grid."""
        sd = np.sqrt(self.effect_cov[:, index, index])
        z = (values[None, :] - self.effect_mean[:, index, None]) / sd[:, None]
        return self.weights @ (np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * sd[:, None]))

    def plane(self, values: np.ndarray) -> np.ndarray:
        """Exact joint density of (β1, log σ) on values × the σ grid."""
        sd = np.sqrt(self.effect_cov[:, 0, 0])
        z = (values[None, :] - self.effect_mean[:, 0, None]) / sd[:, None]
        return self.density[:, None] * np.exp(-0.5 * z**2) / sd[:, None]

    def log_density(self, theta: np.ndarray) -> np.ndarray:
        """Joint log posterior of θ up to a constant, interpolated along the σ grid."""
        spacing = self.log_scales[1] - self.log_scales[0]
        position = np.clip(
            (theta[..., -1] - self.log_scales[0]) / spacing, 0, self.log_scales.size - 1.001
        )
        low = position.astype(int)
        frac = position - low
        log_marginal = np.log(np.maximum(self.density, 1e-300))
        scale_part = (1 - frac) * log_marginal[low] + frac * log_marginal[low + 1]
        mean = (1 - frac)[..., None] * self.effect_mean[low] + frac[..., None] * self.effect_mean[
            low + 1
        ]
        cov = (1 - frac)[..., None, None] * self.effect_cov[low] + frac[
            ..., None, None
        ] * self.effect_cov[low + 1]
        residual = theta[..., :-1] - mean
        solved = np.linalg.solve(cov, residual[..., None])[..., 0]
        _, logdet = np.linalg.slogdet(cov)
        return scale_part - 0.5 * (np.sum(residual * solved, axis=-1) + logdet)


def exact_posterior(history: History, *, half_width: float = 2.0, points: int = 801) -> Exact:
    """Tabulate the exact posterior around the generating σ."""
    grid = math.log(TRUE_SCALE) + np.linspace(-half_width, half_width, points)
    log_lik, effect_mean, effect_cov = _kalman(history, grid)
    log_post = log_lik + _log_scale_prior(grid)
    density = np.exp(log_post - log_post.max())
    density /= density.sum() * (grid[1] - grid[0])
    return Exact(grid, density, effect_mean, effect_cov)


def path_draws(history: History, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exact path draws given each θ (rows), by forward filtering and backward sampling."""
    effects, scale = theta[:, :-1], np.exp(theta[:, -1])
    count, points = theta.shape[0], history.times.size
    means = np.zeros((points, count))
    variances = np.zeros((points, count))
    mean, var = np.zeros(count), np.full(count, INITIAL_SCALE**2)
    for k in range(points):
        if k:
            dt, held = _step(history, k)
            mean = (1 - DECAY * dt) * mean + (effects @ held) * dt
            var = (1 - DECAY * dt) ** 2 * var + scale**2 * dt
        if history.read[k]:
            gain = var / (var + history.reading_sd**2)
            mean = mean + gain * (history.readings[k] - mean)
            var = (1 - gain) * var
        means[k], variances[k] = mean, var
    paths = np.zeros((count, points))
    paths[:, -1] = means[-1] + np.sqrt(variances[-1]) * rng.standard_normal(count)
    for k in range(points - 2, -1, -1):
        dt, held = _step(history, k + 1)
        shrink = 1 - DECAY * dt
        predicted = shrink * means[k] + (effects @ held) * dt
        gain = shrink * variances[k] / (shrink**2 * variances[k] + scale**2 * dt)
        centre = means[k] + gain * (paths[:, k + 1] - predicted)
        spread = np.sqrt(np.maximum((1 - gain * shrink) * variances[k], 0.0))
        paths[:, k] = centre + spread * rng.standard_normal(count)
    return paths


def exact_draws(
    history: History, exact: Exact, count: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Joint posterior draws of θ and the path: σ from the grid, then the effects, then the path."""
    rng = np.random.default_rng(seed)
    spacing = exact.log_scales[1] - exact.log_scales[0]
    index = rng.choice(exact.log_scales.size, size=count, p=exact.weights / exact.weights.sum())
    log_scale = exact.log_scales[index] + rng.uniform(-0.5, 0.5, count) * spacing
    chol = np.linalg.cholesky(exact.effect_cov[index])
    effects = exact.effect_mean[index] + np.einsum(
        "gij,gj->gi", chol, rng.standard_normal((count, chol.shape[1]))
    )
    theta = np.column_stack([effects, log_scale])
    return theta, path_draws(history, theta, rng)


def _log_path_density(history: History, theta: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Log density of a path's Euler–Maruyama steps given θ; broadcasts over leading θ axes."""
    dt = np.diff(history.times)
    drift = -DECAY * path[..., :-1] + theta[..., :-1] @ history.inputs[:, :-1]
    variance = np.exp(2 * theta[..., -1:]) * dt
    residual = np.diff(path, axis=-1) - drift * dt
    initial = -0.5 * (path[..., 0] / INITIAL_SCALE) ** 2
    return initial + np.sum(-0.5 * (np.log(2 * np.pi * variance) + residual**2 / variance), axis=-1)


def _log_prior(theta: np.ndarray) -> np.ndarray:
    return -0.5 * np.sum(theta[..., :-1] ** 2, axis=-1) + _log_scale_prior(theta[..., -1])


@dataclass(frozen=True)
class PathView:
    """What whole paths alone say about θ: p(θ | path) for each path, on the σ grid.

    The readings depend on θ only through the path, so once the path is known they add
    nothing. Given the path, the steps are a regression on the inputs with noise σ²Δt:
    the effects are Gaussian given σ, and σ has a closed form on the grid. Every other
    parameter is integrated out, as it is for a move that holds only the path.
    """

    log_scales: np.ndarray
    weights: np.ndarray
    effect_cov: np.ndarray
    cross: np.ndarray

    def _effect_means(self, index: int) -> np.ndarray:
        """(paths, grid): the mean of one effect given each path and each σ."""
        return self.cross @ self.effect_cov[:, index, :].T / np.exp(2 * self.log_scales)

    @property
    def scale_density(self) -> np.ndarray:
        return self.weights / (self.log_scales[1] - self.log_scales[0])

    @property
    def said(self) -> np.ndarray:
        """(paths, 2): the centre of each path's slice for β1 and for log σ."""
        effect = np.sum(self.weights * self._effect_means(0), axis=1)
        return np.column_stack([effect, self.weights @ self.log_scales])

    def effect_density(self, index: int, values: np.ndarray) -> np.ndarray:
        """(paths, values): the density of one effect given each path."""
        sd = np.sqrt(self.effect_cov[:, index, index])[None, :, None]
        z = (values[None, None, :] - self._effect_means(index)[:, :, None]) / sd
        return np.einsum(
            "pg,pgv->pv", self.weights, np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * sd)
        )


def path_view(history: History, paths: np.ndarray, log_scales: np.ndarray) -> PathView:
    """p(θ | path) for each row of ``paths``."""
    dt = np.diff(history.times)
    held = history.inputs[:, :-1] * dt
    steps = np.diff(paths, axis=-1) + DECAY * paths[..., :-1] * dt
    variance = np.exp(2 * log_scales)
    precision = np.eye(held.shape[0]) + ((held / dt) @ held.T) / variance[:, None, None]
    effect_cov = np.linalg.inv(precision)
    cross = (steps / dt) @ held.T
    fitted = np.einsum("pi,gij,pj->pg", cross, effect_cov, cross) / variance**2
    log_post = (
        _log_scale_prior(log_scales)
        - dt.size * log_scales
        - 0.5 * np.sum(steps**2 / dt, axis=-1)[:, None] / variance
        + 0.5 * fitted
        - 0.5 * np.linalg.slogdet(precision)[1]
    )
    weights = np.exp(log_post - log_post.max(axis=1, keepdims=True))
    return PathView(log_scales, weights / weights.sum(axis=1, keepdims=True), effect_cov, cross)


def draws_given_path(view: PathView, rng: np.random.Generator) -> np.ndarray:
    """One exact θ from p(θ | path) per path: σ from the grid, then the effects given σ."""
    spacing = view.log_scales[1] - view.log_scales[0]
    rows = []
    for weights, cross in zip(view.weights, view.cross, strict=True):
        g = rng.choice(view.log_scales.size, p=weights)
        mean = view.effect_cov[g] @ cross / math.exp(2 * view.log_scales[g])
        effects = rng.multivariate_normal(mean, view.effect_cov[g])
        rows.append(np.append(effects, view.log_scales[g] + rng.uniform(-0.5, 0.5) * spacing))
    return np.array(rows)


def scale_extremes(history: History, exact: Exact, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact paths at the 2.5% and 97.5% points of σ's posterior, effects at their mean given σ."""
    cumulative = np.cumsum(exact.weights) / exact.weights.sum()
    rows = [int(np.searchsorted(cumulative, q)) for q in (0.025, 0.975)]
    theta = np.array([np.append(exact.effect_mean[g], exact.log_scales[g]) for g in rows])
    return theta, path_draws(history, theta, np.random.default_rng(seed))


def step_noise(history: History, theta: np.ndarray, paths: np.ndarray) -> np.ndarray:
    """Each step's noise part, (Δx − drift·Δt)/√Δt, which the model draws from N(0, σ²)."""
    dt = np.diff(history.times)
    drift = -DECAY * paths[..., :-1] + theta[..., :-1] @ history.inputs[:, :-1]
    return (np.diff(paths, axis=-1) - drift * dt) / np.sqrt(dt)


@dataclass(frozen=True)
class Cloud:
    """Exact posterior draws: (β1, log σ), what each draw's path alone says about them, and the paths."""

    shown: np.ndarray
    said: np.ndarray
    paths: np.ndarray


def posterior_cloud(history: History, exact: Exact, count: int, *, seed: int) -> Cloud:
    theta, paths = exact_draws(history, exact, count, seed=seed)
    said = path_view(history, paths, exact.log_scales).said
    return Cloud(theta[:, [0, -1]], said, paths)


@dataclass(frozen=True)
class Panel:
    """One parameter's shadow (given the readings) and one path's slice (given that path)."""

    label: str
    column: int
    grid: np.ndarray
    shadow: np.ndarray
    held: np.ndarray

    @property
    def narrowing(self) -> float:
        return density_sd(self.grid, self.shadow) / density_sd(self.grid, self.held)


def shadow_and_slice(
    history: History, exact: Exact, cloud: Cloud
) -> tuple[int, tuple[Panel, Panel]]:
    """Both parameters' panels, sliced at the draw whose path says the median σ."""
    said = cloud.said[:, 1]
    chosen = int(np.argmin(np.abs(said - np.median(said))))
    view = path_view(history, cloud.paths[chosen][None], exact.log_scales)
    grid = exact.effect_means[0] + np.linspace(-4, 4, 801) * exact.effect_sd[0]
    return chosen, (
        Panel(
            "β1, an effect: loose",
            0,
            grid,
            exact.effect_density(0, grid),
            view.effect_density(0, grid)[0],
        ),
        Panel(
            "log σ, the noise scale: coupled",
            1,
            exact.log_scales,
            exact.density,
            view.scale_density[0],
        ),
    )


@dataclass(frozen=True)
class Walk:
    """An exact sampler's state after each sweep, (chains, sweeps + 1, 2): (β1, log σ) and what the path says."""

    shown: np.ndarray
    said: np.ndarray


def alternating_walk(
    history: History, exact: Exact, *, chains: int, sweeps: int, seed: int
) -> Walk:
    """Exact two-block Gibbs from posterior starts: θ given the path, then the path given θ (FFBS)."""
    rng = np.random.default_rng(seed)
    theta, paths = exact_draws(history, exact, chains, seed=seed)
    view = path_view(history, paths, exact.log_scales)
    shown, said = [theta[:, [0, -1]]], [view.said]
    for _ in range(sweeps):
        theta = draws_given_path(view, rng)
        paths = path_draws(history, theta, rng)
        view = path_view(history, paths, exact.log_scales)
        shown.append(theta[:, [0, -1]])
        said.append(view.said)
    return Walk(np.stack(shown, axis=1), np.stack(said, axis=1))


def integrated_walk(history: History, exact: Exact, *, sweeps: int, seed: int) -> Walk:
    """The ideal of integrating the path out: θ drawn from the shadow each sweep, a fresh path with it."""
    cloud = posterior_cloud(history, exact, sweeps + 1, seed=seed)
    return Walk(cloud.shown[None], cloud.said[None])


def density_sd(grid: np.ndarray, density: np.ndarray) -> float:
    """Standard deviation of a density tabulated on a uniform grid."""
    weights = density * (grid[1] - grid[0])
    mean = weights @ grid
    return float(np.sqrt(weights @ (grid - mean) ** 2))


@dataclass(frozen=True)
class Votes:
    """How often a proposed θ′ wins, against its distance from θ in posterior SDs."""

    distances: np.ndarray
    one_path: np.ndarray
    integrated: np.ndarray


def votes(
    history: History, exact: Exact, distances: np.ndarray, *, coordinate: int, draws: int = 400
) -> Votes:
    """Barker's choice between θ and θ′, judged by the current path or with the path integrated out.

    ``one_path`` keeps today's path: θ′ wins with probability p(θ′, x) / (p(θ, x) +
    p(θ′, x)). ``integrated`` uses p(θ | readings) instead, which is what the label
    move reaches when its smoother proposes paths suited to θ′.
    """
    theta, paths = exact_draws(history, exact, draws)
    covariance = np.diag(np.append(exact.effect_sd**2, exact.scale_sd**2))
    sd = math.sqrt(covariance[coordinate, coordinate])
    held = _log_prior(theta) + _log_path_density(history, theta, paths)
    marginal = exact.log_density(theta)
    one_path, integrated = [], []
    for distance in distances:
        moved = theta.copy()
        moved[:, coordinate] += distance * sd
        joint = _log_prior(moved) + _log_path_density(history, moved, paths)
        one_path.append(np.mean(1 / (1 + np.exp(held - joint))))
        integrated.append(np.mean(1 / (1 + np.exp(marginal - exact.log_density(moved)))))
    return Votes(distances, np.array(one_path), np.array(integrated))


def effective_per_thousand(values: np.ndarray, ess: float) -> float:
    """Effective draws per 1,000 retained draws."""
    return 1000 * ess / values.size


def autocorrelation(values: np.ndarray, lags: int) -> np.ndarray:
    """Mean over chains of each chain's autocorrelation, lags 0..``lags``."""
    curves = []
    for chain in np.asarray(values, dtype=float):
        centred = chain - chain.mean()
        spectrum = np.fft.rfft(centred, n=2 * centred.size)
        full = np.fft.irfft(spectrum * np.conj(spectrum))[: centred.size]
        curves.append(full[: lags + 1] / full[0])
    return np.mean(curves, axis=0)


def sweeps_per_independent_draw(values: np.ndarray) -> float:
    """Integrated autocorrelation time of (chains, sweeps), by Geyer's initial positive sequence."""
    centred = values - values.mean()
    n = centred.shape[1]
    spectrum = np.fft.rfft(centred, n=2 * n, axis=1)
    covariance = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n].mean(axis=0)
    rho = covariance / covariance[0]
    pairs = rho[: n - 1 : 2] + rho[1:n:2]
    stop = int(np.argmax(pairs <= 0)) if np.any(pairs <= 0) else pairs.size
    return float(-1 + 2 * pairs[:stop].sum())


# ── The reference picture: θ across, the whole path squeezed to one axis up ────────


@dataclass(frozen=True)
class Shape:
    """One parameter's reference picture: a standard bivariate normal in (θ, x).

    x is the whole path squeezed to one axis. Given x, θ is ``narrowing`` times narrower
    than alone, the factor Part 0 measures on the real model. The shadow (θ alone) is
    N(0, 1), the slice at x (θ given x) is N(ρx, w²) and the band at θ (x given θ) is
    N(ρθ, w²), where w = 1 / narrowing.
    """

    label: str
    narrowing: float

    @property
    def width(self) -> float:
        return 1 / self.narrowing

    @property
    def rho(self) -> float:
        return math.sqrt(1 - self.width**2)

    def log_density(self, q: np.ndarray) -> float:
        return float(-0.5 * (q[0] ** 2 - 2 * self.rho * q[0] * q[1] + q[1] ** 2) / self.width**2)

    def gradient(self, q: np.ndarray) -> np.ndarray:
        return -np.array([q[0] - self.rho * q[1], q[1] - self.rho * q[0]]) / self.width**2


@dataclass(frozen=True)
class Moves:
    """The textbook move of each strategy on one shape, from the same start."""

    alternate: np.ndarray
    integrated: np.ndarray
    rejected: np.ndarray
    trajectories: tuple[np.ndarray, ...]


def textbook_moves(shape: Shape, start: tuple[float, float], *, seed: int) -> Moves:
    """Each strategy's textbook move on one shape, exact for its bivariate normal.

    - alternate: θ given x, then x given θ, 15 times;
    - integrate the path out: 20 random-walk θ′ accepted on the shadow N(0, 1), each with
      a fresh x from the band at θ′;
    - move both: 4 leapfrog trajectories with unit mass (the posterior's diagonal, as NUTS
      adapts it), a step of a quarter of the narrow scale so the curve draws smoothly,
      and a quarter turn along the long axis where NUTS would choose each length itself.
    """
    rng = np.random.default_rng(seed)
    theta, x = start
    alternate = [(theta, x)]
    for _ in range(15):
        theta = shape.rho * x + shape.width * rng.standard_normal()
        alternate.append((theta, x))
        x = shape.rho * theta + shape.width * rng.standard_normal()
        alternate.append((theta, x))
    theta, x = start
    integrated, rejected = [(theta, x)], []
    for _ in range(20):
        proposal = theta + 2.0 * rng.standard_normal()
        fresh = shape.rho * proposal + shape.width * rng.standard_normal()
        if math.log(rng.uniform()) < 0.5 * (theta**2 - proposal**2):
            theta, x = proposal, fresh
            integrated.append((theta, x))
        else:
            rejected.append((proposal, fresh))
    step = 0.25 * math.sqrt(1 - shape.rho)
    steps = round(0.5 * math.pi * math.sqrt(1 + shape.rho) / step)
    q = np.asarray(start, dtype=float)
    trajectories = []
    for _ in range(4):
        p = rng.standard_normal(2)
        start_energy = -shape.log_density(q) + 0.5 * p @ p
        moved, momentum, points = q.copy(), p.copy(), [q.copy()]
        for _ in range(steps):
            momentum = momentum + 0.5 * step * shape.gradient(moved)
            moved = moved + step * momentum
            momentum = momentum + 0.5 * step * shape.gradient(moved)
            points.append(moved.copy())
        if (
            math.log(rng.uniform())
            < start_energy + shape.log_density(moved) - 0.5 * momentum @ momentum
        ):
            q = moved
        trajectories.append(np.array(points))
    return Moves(np.array(alternate), np.array(integrated), np.array(rejected), tuple(trajectories))


# ── Figures ─────────────────────────────────────────────────────────────────────────
# Validated categorical slots 1, 2, 3 and 7: one colour per sampler piece, everywhere.

C_JOINT = "#2a78d6"
C_LABEL = "#eb6834"
C_SMOOTH = "#1baf7a"
C_CONDITIONAL = "#6250d6"
C_INK = "#0b0b0b"
C_INK_2 = "#52514e"
C_MUTED = "#9aa0a6"
C_GRID = "#e6e5e1"
C_FILL = "rgba(82, 81, 78, 0.16)"


def _layout(fig: go.Figure, *, title: str, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 15, "color": C_INK},
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        height=height,
        margin={"l": 64, "r": 24, "t": 118 if legend else 84, "b": 56},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, system-ui, sans-serif", "size": 12, "color": C_INK_2},
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1 + 40 / max(height - 174, 1),
            "x": 0,
            "font": {"size": 12},
        },
        hovermode="closest",
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=C_GRID, zeroline=False, linecolor=C_MUTED, ticks="outside"
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=C_GRID, zeroline=False, linecolor=C_MUTED, ticks="outside"
    )
    return fig


def _dedupe_legend(fig: go.Figure) -> go.Figure:
    seen: set[str] = set()
    for trace in fig.data:
        if trace.name in seen:
            trace.showlegend = False
        elif trace.name:
            seen.add(trace.name)
    return fig


def _band(
    fig: go.Figure,
    xs: np.ndarray,
    low: float | np.ndarray,
    high: float | np.ndarray,
    *,
    row: int,
    col: int,
    name: str,
) -> None:
    lows = np.broadcast_to(low, xs.shape)
    highs = np.broadcast_to(high, xs.shape)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([highs, lows[::-1]]),
            fill="toself",
            fillcolor=C_FILL,
            line={"width": 0},
            name=name,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def fig_history(history: History, *, title: str) -> go.Figure:
    """The readings of x over eight days, and the five inputs that push it."""
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.68, 0.32], shared_xaxes=True, vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(
            x=history.times[history.read],
            y=history.readings[history.read],
            mode="markers",
            marker={"color": C_INK_2, "size": 5},
            name="readings of x",
        ),
        row=1,
        col=1,
    )
    for j in range(history.inputs.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=history.times,
                y=history.inputs[j] * 0.8 + j,
                mode="lines",
                line={"color": C_MUTED, "width": 1.5, "shape": "hv"},
                name=f"u{j + 1} (true effect {TRUE_EFFECTS[j]:+.1f})",
                showlegend=False,
                hovertemplate=f"u{j + 1}, true effect {TRUE_EFFECTS[j]:+.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_annotation(
            text=f"u{j + 1}: β = {TRUE_EFFECTS[j]:+.1f}",
            x=history.times[-1],
            y=j + 0.4,
            xref="x2",
            yref="y2",
            xanchor="left",
            showarrow=False,
            font={"size": 11, "color": C_INK_2},
        )
    fig.update_yaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text="inputs (on/off)", row=2, col=1)
    fig.update_xaxes(title_text="day", row=2, col=1)
    fig.update_layout(margin={"r": 110})
    return _layout(fig, title=title, height=440)


def _corner_note(fig: go.Figure, text: str, *, axis: int) -> None:
    fig.add_annotation(
        text=text,
        xref=f"x{axis if axis > 1 else ''} domain",
        yref=f"y{axis if axis > 1 else ''} domain",
        x=0.02,
        y=0.96,
        showarrow=False,
        xanchor="left",
        align="left",
        font={"size": 12, "color": C_INK},
    )


def fig_what_paths_know(
    history: History, theta: np.ndarray, paths: np.ndarray, *, start: float = 3.0, days: float = 1.0
) -> go.Figure:
    """Two exact paths, σ at the low and high end of its posterior: same readings, different steps."""
    window = (history.times >= start) & (history.times <= start + days)
    noise = step_noise(history, theta, paths)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.1,
        subplot_titles=[
            f"day {start:.0f} to {start + days:.0f}: the readings and two paths",
            f"the noise in each path's {noise.shape[1]} steps",
        ],
    )
    styles = ((C_MUTED, 3.5, "low"), (C_INK, 1.4, "high"))
    for (colour, width, end), path, steps, log_scale in zip(
        styles, paths, noise, theta[:, -1], strict=True
    ):
        name = f"σ = {math.exp(log_scale):.2f}, the {end} end of σ's posterior"
        fig.add_trace(
            go.Scatter(
                x=history.times[window],
                y=path[window],
                mode="lines",
                line={"color": colour, "width": width},
                name=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=steps,
                xbins={"start": -1.6, "end": 1.6, "size": 0.08},
                marker={"color": colour, "line": {"color": "white", "width": 1}},
                opacity=0.75,
                name=name,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=history.times[window],
            y=history.readings[window],
            mode="markers",
            marker={"color": C_INK_2, "size": 7, "symbol": "circle-open", "line": {"width": 1.5}},
            name=f"readings (noise SD {history.reading_sd:.1f})",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="day", row=1, col=1)
    fig.update_yaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="(Δx − drift·Δt) / √Δt", row=1, col=2)
    fig.update_yaxes(title_text="steps", row=1, col=2)
    fig.update_annotations(font={"size": 12, "color": C_INK})
    return _dedupe_legend(
        _layout(fig, title="The readings can't tell these paths apart; their steps can", height=420)
    )


SPAN = 3.2


def _reference(fig: go.Figure, shape: Shape, *, row: int, col: int, shadow: bool) -> None:
    """One parameter's reference: 1, 2 and 3 SD contours, and its shadow along the bottom."""
    grid = np.linspace(-SPAN, SPAN, 161)
    theta, x = np.meshgrid(grid, grid)
    fig.add_trace(
        go.Contour(
            x=grid,
            y=grid,
            z=np.sqrt(theta**2 - 2 * shape.rho * theta * x + x**2) / shape.width,
            showscale=False,
            contours={"coloring": "lines", "start": 1, "end": 3, "size": 1},
            line={"width": 1},
            colorscale=[[0, C_INK_2], [1, C_MUTED]],
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    if shadow:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([grid, grid[::-1]]),
                y=np.concatenate([-SPAN + 0.9 * np.exp(-0.5 * grid**2), np.full(grid.size, -SPAN)]),
                fill="toself",
                fillcolor=C_FILL,
                line={"color": C_INK_2, "width": 1},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_xaxes(
        range=[-SPAN, SPAN],
        showticklabels=False,
        zeroline=False,
        title_text=shape.label.split(",")[0],
        row=row,
        col=col,
    )
    fig.update_yaxes(range=[-SPAN, SPAN], showticklabels=False, zeroline=False, row=row, col=col)


def _segment(
    fig: go.Figure,
    xs: Sequence[float] | np.ndarray,
    ys: Sequence[float] | np.ndarray,
    colour: str,
    *,
    row: int,
    col: int,
    width: float = 2,
    dash: str = "solid",
    opacity: float = 1.0,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=list(xs),
            y=list(ys),
            mode="lines",
            line={"color": colour, "width": width, "dash": dash},
            opacity=opacity,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _arrow(
    fig: go.Figure,
    tail: Sequence[float],
    head: Sequence[float],
    colour: str,
    *,
    axis: int,
    dash: bool = False,
) -> None:
    suffix = "" if axis == 1 else str(axis)
    fig.add_annotation(
        x=head[0],
        y=head[1],
        ax=tail[0],
        ay=tail[1],
        xref=f"x{suffix}",
        yref=f"y{suffix}",
        axref=f"x{suffix}",
        ayref=f"y{suffix}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=colour,
        text="",
        opacity=0.6 if dash else 1.0,
    )


def _label(
    fig: go.Figure, at: Sequence[float], text: str, *, axis: int, anchor: str = "left"
) -> None:
    suffix = "" if axis == 1 else str(axis)
    fig.add_annotation(
        x=at[0],
        y=at[1],
        xref=f"x{suffix}",
        yref=f"y{suffix}",
        text=text,
        showarrow=False,
        xanchor=anchor,
        align=anchor,
        font={"size": 11, "color": C_INK},
        bgcolor="rgba(255, 255, 255, 0.75)",
    )


def _mark_state(fig: go.Figure, state: Sequence[float], *, row: int, col: int) -> None:
    fig.add_trace(
        go.Scatter(
            x=[state[0]],
            y=[state[1]],
            mode="markers",
            marker={"size": 11, "color": "white", "line": {"width": 2.5, "color": C_INK}},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _slice_bar(fig: go.Figure, shape: Shape, state: Sequence[float], *, row: int, col: int) -> None:
    """θ given the path at ``state``: the purple horizontal range, ±2 SD."""
    centre = shape.rho * state[1]
    _segment(
        fig,
        [centre - 2 * shape.width, centre + 2 * shape.width],
        [state[1], state[1]],
        C_CONDITIONAL,
        row=row,
        col=col,
        width=7,
        opacity=0.35,
    )


def _band_bar(
    fig: go.Figure, shape: Shape, theta: float, colour: str, *, row: int, col: int
) -> None:
    """The path given θ: the vertical range at ``theta``, ±2 SD."""
    centre = shape.rho * theta
    _segment(
        fig,
        [theta, theta],
        [centre - 2 * shape.width, centre + 2 * shape.width],
        colour,
        row=row,
        col=col,
        width=7,
        opacity=0.35,
    )


def start_state(shape: Shape, theta: float = -1.2) -> tuple[float, float]:
    """The shared starting point: θ in the shadow's left tail, the path at its band's centre."""
    return theta, shape.rho * theta


def fig_reference(shapes: Sequence[Shape]) -> go.Figure:
    """The reference: each shape with its shadow, and the slice and band through one state."""
    fig = make_subplots(
        rows=1,
        cols=len(shapes),
        subplot_titles=[shape.label for shape in shapes],
        horizontal_spacing=0.06,
    )
    for col, shape in enumerate(shapes, start=1):
        state = start_state(shape)
        _reference(fig, shape, row=1, col=col, shadow=True)
        _slice_bar(fig, shape, state, row=1, col=col)
        _band_bar(fig, shape, state[0], C_SMOOTH, row=1, col=col)
        _mark_state(fig, state, row=1, col=col)
        _label(fig, (-SPAN + 0.15, -SPAN + 0.55), "shadow: θ given the readings", axis=col)
        _label(
            fig,
            (shape.rho * state[1] + 2 * shape.width + 0.1, state[1]),
            "slice: θ given this path",
            axis=col,
        )
        _label(
            fig,
            (state[0] + 0.12, shape.rho * state[0] + 2 * shape.width - 0.1),
            "band: the path given this θ",
            axis=col,
        )
    fig.update_yaxes(title_text="x: the whole path, on one axis", row=1, col=1)
    fig.update_annotations(font={"size": 12})
    return _layout(
        fig, title="The reference: each parameter against the whole path", height=440, legend=False
    )


def fig_strategy_overview(shapes: Sequence[Shape], moves: Sequence[Moves]) -> go.Figure:
    """Each strategy's textbook move on both shapes: one row per strategy."""
    fig = make_subplots(
        rows=3,
        cols=len(shapes),
        column_titles=[shape.label for shape in shapes],
        row_titles=["alternate", "integrate the path out", "move both"],
        horizontal_spacing=0.05,
        vertical_spacing=0.06,
    )
    for col, (shape, move) in enumerate(zip(shapes, moves, strict=True), start=1):
        for row in (1, 2, 3):
            _reference(fig, shape, row=row, col=col, shadow=row == 2)
            if row < 3:
                fig.update_xaxes(title_text="", row=row, col=col)
        points = move.alternate
        for k in range(0, points.shape[0] - 1, 2):
            _segment(fig, points[k : k + 2, 0], points[k : k + 2, 1], C_CONDITIONAL, row=1, col=col)
            _segment(
                fig, points[k + 1 : k + 3, 0], points[k + 1 : k + 3, 1], C_SMOOTH, row=1, col=col
            )
        if move.rejected.size:
            fig.add_trace(
                go.Scatter(
                    x=move.rejected[:, 0],
                    y=move.rejected[:, 1],
                    mode="markers",
                    marker={
                        "color": C_LABEL,
                        "size": 8,
                        "symbol": "x-thin",
                        "line": {"width": 2, "color": C_LABEL},
                    },
                    opacity=0.5,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=move.integrated[:, 0],
                y=move.integrated[:, 1],
                mode="lines+markers",
                line={"color": C_LABEL, "width": 1.5},
                marker={"size": 6},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=col,
        )
        for trajectory in move.trajectories:
            _segment(fig, trajectory[:, 0], trajectory[:, 1], C_JOINT, row=3, col=col, width=1.5)
        for row in (1, 2, 3):
            _mark_state(fig, move.alternate[0], row=row, col=col)
    fig.update_yaxes(title_text="the whole path", col=1)
    fig.update_annotations(font={"size": 12})
    return _layout(
        fig,
        title="The three strategies on the reference, each at its textbook best",
        height=900,
        legend=False,
    )


def _along(start: Sequence[float], along: float, across: float, turns: float) -> np.ndarray:
    """A schematic trajectory: ``along`` up the diagonal, swinging ``across`` it ``turns`` times."""
    t = np.linspace(0, 1, 160)[:, None]
    up = np.array([1.0, 1.0]) / math.sqrt(2)
    side = np.array([1.0, -1.0]) / math.sqrt(2)
    return np.asarray(start) + along * t * up + across * np.sin(2 * math.pi * turns * t) * side


def fig_joint_schematic(shapes: Sequence[Shape]) -> go.Figure:
    """Part 1 on the reference: how far one joint NUTS draw travels in each chart (schematic)."""
    fig = make_subplots(
        rows=1, cols=len(shapes), subplot_titles=[s.label for s in shapes], horizontal_spacing=0.06
    )
    for col, shape in enumerate(shapes, start=1):
        state = start_state(shape)
        _reference(fig, shape, row=1, col=col, shadow=False)
        if shape.narrowing < 2:
            curve = _along(state, 2.6, 0.5, 1.5)
            _segment(fig, curve[:, 0], curve[:, 1], C_JOINT, row=1, col=col)
            _label(
                fig,
                curve[-1] + np.array([-1.2, 0.35]),
                "either chart: one draw<br>crosses the cloud",
                axis=col,
            )
        else:
            long = _along(state, 2.8, 0.1, 3)
            short = _along(state, 0.6, 0.06, 1)
            _segment(fig, long[:, 0], long[:, 1], C_JOINT, row=1, col=col)
            _segment(fig, short[:, 0], short[:, 1], C_INK, row=1, col=col, width=4)
            _arrow(fig, short[-8], short[-1], C_INK, axis=col)
            _label(
                fig,
                long[-1] + np.array([-1.7, 0.3]),
                "innovation chart: one draw<br>travels along the ridge",
                axis=col,
            )
            _label(
                fig,
                short[-1] + np.array([0.15, -0.45]),
                "centred chart: one draw<br>moves only locally",
                axis=col,
            )
        _mark_state(fig, state, row=1, col=col)
    fig.update_yaxes(title_text="x: the whole path, on one axis", row=1, col=1)
    fig.update_annotations(font={"size": 12})
    return _layout(
        fig,
        title="Part 1 on the reference: one joint NUTS draw (schematic)",
        height=440,
        legend=False,
    )


def fig_label_schematic(shapes: Sequence[Shape], *, step: float = 1.5) -> go.Figure:
    """Part 2 on the reference: one label move, from θ to θ′ (schematic)."""
    fig = make_subplots(
        rows=1, cols=len(shapes), subplot_titles=[s.label for s in shapes], horizontal_spacing=0.06
    )
    for col, shape in enumerate(shapes, start=1):
        state = start_state(shape)
        proposal = state[0] + step
        _reference(fig, shape, row=1, col=col, shadow=False)
        _band_bar(fig, shape, state[0], C_MUTED, row=1, col=col)
        _band_bar(fig, shape, proposal, C_LABEL, row=1, col=col)
        _segment(
            fig, [proposal, proposal], [-SPAN, SPAN], C_LABEL, row=1, col=col, width=1, dash="dot"
        )
        quantiles = np.array([-1.5, -0.75, 0.0, 0.75, 1.5]) * shape.width
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([np.full(5, state[0]), np.full(5, proposal)]),
                y=np.concatenate(
                    [shape.rho * state[0] + quantiles, shape.rho * proposal + quantiles]
                ),
                mode="markers",
                marker={"size": 6, "color": np.repeat([C_INK_2, C_LABEL], 5).tolist()},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        landed = (proposal, shape.rho * proposal + 0.4 * shape.width)
        _arrow(fig, state, landed, C_LABEL, axis=col)
        _mark_state(fig, state, row=1, col=col)
        if shape.narrowing < 2:
            _label(
                fig,
                (proposal + 0.2, landed[1] + 0.9),
                "the two bands overlap: any new<br>path suits θ and θ′ alike",
                axis=col,
            )
        else:
            stayed = (state[0], shape.rho * state[0] - 1.8 * shape.width)
            _arrow(fig, state, stayed, C_INK_2, axis=col, dash=True)
            _label(
                fig,
                (proposal + 0.15, landed[1]),
                "a new path lands in θ′'s band:<br>θ′ wins",
                axis=col,
            )
            _label(
                fig,
                (stayed[0] + 0.15, stayed[1] - 0.25),
                "it stays in θ's band: θ is kept,<br>though the path still changed",
                axis=col,
            )
        _label(fig, (state[0] - 0.1, SPAN - 0.25), "θ", axis=col, anchor="right")
        _label(fig, (proposal + 0.1, SPAN - 0.25), "θ′", axis=col)
    fig.update_yaxes(title_text="x: the whole path, on one axis", row=1, col=1)
    fig.update_annotations(font={"size": 12})
    return _layout(
        fig, title="Part 2 on the reference: one label move (schematic)", height=440, legend=False
    )


def _smoother_draws(
    fig: go.Figure,
    history: History,
    band: tuple[np.ndarray, np.ndarray],
    run: FitResult,
    colour: str,
    *,
    row: int,
    col: int,
    draws: int = 6,
    until: float = 2.5,
) -> None:
    shown = history.times <= until
    times = history.times[shown]
    _band(
        fig, times, band[0][shown], band[1][shown], row=row, col=col, name="exact 90% band of paths"
    )
    for s in range(draws):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=run.paths[0, s, shown],
                mode="lines",
                line={"color": colour, "width": 1.3},
                opacity=0.35 + 0.65 * s / max(draws - 1, 1),
                name=f"{draws} consecutive draws",
            ),
            row=row,
            col=col,
        )
    read = history.read[shown]
    fig.add_trace(
        go.Scatter(
            x=times[read],
            y=history.readings[shown][read],
            mode="markers",
            marker={"color": C_INK, "size": 4},
            name="readings",
        ),
        row=row,
        col=col,
    )


def fig_smoother_schematic(
    shapes: Sequence[Shape],
    history: History,
    band: tuple[np.ndarray, np.ndarray],
    runs: Mapping[str, FitResult],
) -> go.Figure:
    """Part 3: the smoother's move on the reference (schematic), and its real path draws over time."""
    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.16,
        horizontal_spacing=0.06,
        subplot_titles=[s.label for s in shapes]
        + [f"real draws at the true θ: {label}" for label in runs],
    )
    for col, shape in enumerate(shapes, start=1):
        state = start_state(shape)
        _reference(fig, shape, row=1, col=col, shadow=False)
        _band_bar(fig, shape, state[0], C_SMOOTH, row=1, col=col)
        for q in np.array([-1.6, -1.0, -0.4, 0.4, 1.0, 1.6]) * shape.width:
            _arrow(fig, state, (state[0], shape.rho * state[0] + q), C_SMOOTH, axis=col)
        _mark_state(fig, state, row=1, col=col)
        _label(
            fig,
            (state[0] + 0.15, shape.rho * state[0] + 2 * shape.width + 0.15),
            "θ fixed: new paths only<br>within the band",
            axis=col,
        )
    for col, (run, colour) in enumerate(
        zip(runs.values(), (C_SMOOTH, C_MUTED), strict=True), start=1
    ):
        _smoother_draws(fig, history, band, run, colour, row=2, col=col)
        fig.update_xaxes(title_text="day", row=2, col=col)
    fig.update_yaxes(title_text="x: the whole path, on one axis", row=1, col=1)
    fig.update_yaxes(title_text="x(t)", row=2, col=1)
    fig.update_annotations(font={"size": 12})
    for trace in fig.data:
        trace.showlegend = False
    return _layout(
        fig,
        title="Part 3 on the reference: the path given θ (schematic, top) and real draws (bottom)",
        height=760,
        legend=False,
    )


def _span(panel: Panel) -> list[float]:
    centre = float(panel.grid[np.argmax(panel.shadow)])
    spread = density_sd(panel.grid, panel.shadow)
    return [centre - 4 * spread, centre + 4 * spread]


def _cloud_trace(cloud: Cloud, column: int, *, legend: bool = True) -> go.Scatter:
    return go.Scatter(
        x=cloud.shown[:, column],
        y=cloud.said[:, column],
        mode="markers",
        marker={"color": C_MUTED, "size": 3, "opacity": 0.45},
        name="exact posterior draws (θ, path)",
        showlegend=legend,
        hoverinfo="skip",
    )


def fig_shadow_slice(cloud: Cloud, chosen: int, panels: Sequence[Panel]) -> go.Figure:
    """The joint posterior of each parameter and what the path says about it: shadow and slice."""
    fig = make_subplots(
        rows=2,
        cols=len(panels),
        shared_xaxes=True,
        row_heights=[0.64, 0.36],
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        subplot_titles=[panel.label for panel in panels],
    )
    for col, panel in enumerate(panels, start=1):
        span = _span(panel)
        said = cloud.said[:, panel.column]
        level = float(said[chosen])
        band = np.abs(said - level) < 0.02 * (span[1] - span[0])
        fig.add_trace(_cloud_trace(cloud, panel.column), row=1, col=col)
        fig.add_trace(
            go.Scatter(
                x=span,
                y=[level, level],
                mode="lines",
                line={"color": C_CONDITIONAL, "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=cloud.shown[band, panel.column],
                y=said[band],
                mode="markers",
                marker={"color": C_CONDITIONAL, "size": 5},
                name="draws whose paths say the same: a slice",
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=panel.grid,
                y=panel.shadow,
                mode="lines",
                fill="tozeroy",
                fillcolor=C_FILL,
                line={"color": C_INK_2, "width": 1.5},
                name="given the readings: the shadow",
            ),
            row=2,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=panel.grid,
                y=panel.held,
                mode="lines",
                line={"color": C_CONDITIONAL, "width": 2},
                name="given that one path: its slice",
            ),
            row=2,
            col=col,
        )
        _corner_note(
            fig, f"slice {panel.narrowing:.2f}× narrower<br>than the shadow", axis=len(panels) + col
        )
        fig.update_xaxes(range=span, row=1, col=col)
        fig.update_yaxes(range=span, row=1, col=col)
        fig.update_xaxes(range=span, title_text=panel.label.split(",")[0], row=2, col=col)
        fig.update_yaxes(showticklabels=False, row=2, col=col)
    fig.update_yaxes(title_text="what the path alone says", row=1, col=1)
    fig.update_annotations(font={"size": 12, "color": C_INK})
    return _dedupe_legend(
        _layout(fig, title="The whole posterior, seen through one parameter (exact)", height=640)
    )


def fig_strategies(
    alternating: Walk,
    integrated: Walk,
    cloud: Cloud,
    panels: Sequence[Panel],
    *,
    plane_sweeps: int = 25,
    trace_sweeps: int = 300,
) -> go.Figure:
    """Two exact samplers on log σ's ridge, then both over many sweeps for both parameters."""
    effect, scale = panels
    taus = [sweeps_per_independent_draw(alternating.shown[:, 1:, p.column]) for p in panels]
    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.56, 0.44],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
        subplot_titles=[
            f"alternate: {plane_sweeps} sweeps on the log σ ridge",
            f"integrate the path out: {plane_sweeps} sweeps on the ridge",
            f"log σ · alternate needs {taus[1]:.0f} sweeps per independent draw",
            f"β1 · alternate needs {taus[0]:.1f} sweeps per independent draw",
        ],
    )
    xs = alternating.shown[0, : plane_sweeps + 1, scale.column]
    ys = alternating.said[0, : plane_sweeps + 1, scale.column]
    across: tuple[list[float | None], list[float | None]] = ([], [])
    up: tuple[list[float | None], list[float | None]] = ([], [])
    for s in range(1, xs.size):
        across[0].extend([xs[s - 1], xs[s], None])
        across[1].extend([ys[s - 1], ys[s - 1], None])
        up[0].extend([xs[s], xs[s], None])
        up[1].extend([ys[s - 1], ys[s], None])
    fig.add_trace(_cloud_trace(cloud, scale.column, legend=False), row=1, col=1)
    for (x, y), colour, name in (
        (across, C_CONDITIONAL, "θ given the path (exact conditional draw)"),
        (up, C_SMOOTH, "the path given θ (the smoother's move)"),
    ):
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", line={"color": colour, "width": 2}, name=name),
            row=1,
            col=1,
        )
    fig.add_trace(_cloud_trace(cloud, scale.column, legend=False), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=integrated.shown[0, : plane_sweeps + 1, scale.column],
            y=integrated.said[0, : plane_sweeps + 1, scale.column],
            mode="lines+markers",
            line={"color": C_LABEL, "width": 1},
            marker={"size": 5},
            name="θ with a fresh path (the label move's ideal)",
        ),
        row=1,
        col=2,
    )
    for col in (1, 2):
        fig.update_xaxes(range=_span(scale), title_text="log σ", row=1, col=col)
        fig.update_yaxes(range=_span(scale), row=1, col=col)
    sweeps = np.arange(trace_sweeps + 1)
    for col, panel in ((1, scale), (2, effect)):
        mean = float(panel.grid @ panel.shadow / panel.shadow.sum())
        spread = density_sd(panel.grid, panel.shadow)
        _band(fig, sweeps, mean - 2 * spread, mean + 2 * spread, row=2, col=col, name="")
        fig.data[-1].showlegend = False
        for walk, colour in ((integrated, C_LABEL), (alternating, C_CONDITIONAL)):
            fig.add_trace(
                go.Scatter(
                    x=sweeps,
                    y=walk.shown[0, : trace_sweeps + 1, panel.column],
                    mode="lines",
                    line={"color": colour, "width": 1.3},
                    showlegend=False,
                ),
                row=2,
                col=col,
            )
        fig.update_xaxes(title_text="sweep", row=2, col=col)
        fig.update_yaxes(range=_span(panel), title_text=panel.label.split(",")[0], row=2, col=col)
    fig.update_yaxes(title_text="log σ said by the path alone", row=1, col=1)
    fig.update_annotations(font={"size": 12, "color": C_INK})
    return _dedupe_legend(
        _layout(
            fig, title="Alternate, or integrate the path out, each at its best (exact)", height=780
        )
    )


def fig_ridge_runs(
    cloud: Cloud,
    history: History,
    log_scales: np.ndarray,
    runs: Mapping[str, tuple[FitResult, str]],
    *,
    sweeps: int = 30,
) -> go.Figure:
    """One chain per real fit on Part 0's log σ ridge: consecutive retained draws, joined."""
    titles = [
        f"{label}<br>log σ: {effective_per_thousand(run.column('sigma'), run.ess['sigma']):.0f} effective per 1,000"
        for label, (run, _) in runs.items()
    ]
    fig = make_subplots(rows=1, cols=len(runs), subplot_titles=titles, horizontal_spacing=0.06)
    centre, spread = float(np.mean(cloud.shown[:, 1])), float(np.std(cloud.shown[:, 1]))
    span = [centre - 4 * spread, centre + 4 * spread]
    for col, (label, (run, colour)) in enumerate(runs.items(), start=1):
        said = path_view(history, run.paths[0, :sweeps], log_scales).said[:, 1]
        fig.add_trace(_cloud_trace(cloud, 1), row=1, col=col)
        fig.add_trace(
            go.Scatter(
                x=run.column("sigma")[0, :sweeps],
                y=said,
                mode="lines+markers",
                line={"color": colour, "width": 1.2},
                marker={"size": 6, "color": colour},
                name=label,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(range=span, title_text="log σ", row=1, col=col)
        fig.update_yaxes(range=span, row=1, col=col)
    fig.update_yaxes(title_text="log σ said by the path alone", row=1, col=1)
    fig.update_annotations(font={"size": 12, "color": C_INK})
    return _layout(
        fig,
        title=f"The first {sweeps} retained draws of one chain, on the log σ ridge (real fits)",
        height=430,
        legend=False,
    )


def fig_pin(rows: Sequence[tuple[int, Sequence[Panel]]]) -> go.Figure:
    """Same readings, finer grid: log σ's slice narrows, β1's doesn't (exact).

    Each row is (steps, panels), with the panels from `shadow_and_slice`.
    """
    fig = make_subplots(
        rows=2,
        cols=len(rows),
        subplot_titles=[f"log σ · {steps} steps" for steps, _ in rows]
        + [f"β1 · {steps} steps" for steps, _ in rows],
        vertical_spacing=0.2,
        horizontal_spacing=0.06,
    )
    for col, (_, panels) in enumerate(rows, start=1):
        for row, panel in enumerate(reversed(panels), start=1):
            fig.add_trace(
                go.Scatter(
                    x=panel.grid,
                    y=panel.shadow,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=C_FILL,
                    line={"color": C_INK_2, "width": 1.5},
                    name="given the readings: the shadow",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=panel.grid,
                    y=panel.held,
                    mode="lines",
                    line={"color": C_CONDITIONAL, "width": 2},
                    name="given one whole path: its slice",
                ),
                row=row,
                col=col,
            )
            _corner_note(
                fig, f"slice {panel.narrowing:.2f}× narrower", axis=(row - 1) * len(rows) + col
            )
            fig.update_xaxes(range=_span(panel), row=row, col=col)
            fig.update_yaxes(showticklabels=False, row=row, col=col)
    fig.update_annotations(font={"size": 12, "color": C_INK})
    return _dedupe_legend(
        _layout(
            fig,
            title="Same readings on a finer grid: σ's slice narrows, β1's doesn't (exact)",
            height=560,
        )
    )


def fig_joint_runs(runs: Mapping[str, FitResult], exact: Exact) -> go.Figure:
    """Real joint NUTS fits per chart: log σ per chain against the exact posterior."""
    names = {"centred": "centred", "non_centred": "innovation"}
    titles = [
        f"{names[chart]}: {effective_per_thousand(run.column('sigma'), run.ess['sigma']):.0f} effective per 1,000 draws, "
        f"{np.median(run.extra['num_steps']):.0f} gradients per draw"
        for chart, run in runs.items()
    ]
    fig = make_subplots(rows=1, cols=len(runs), subplot_titles=titles, horizontal_spacing=0.07)
    for col, run in enumerate(runs.values(), start=1):
        sigma = run.column("sigma")
        draws = np.arange(sigma.shape[1])
        _band(
            fig,
            draws,
            exact.scale_mean - 2 * exact.scale_sd,
            exact.scale_mean + 2 * exact.scale_sd,
            row=1,
            col=col,
            name="exact posterior ±2 SD",
        )
        for chain in sigma:
            fig.add_trace(
                go.Scatter(
                    x=draws,
                    y=chain,
                    mode="lines",
                    line={"color": C_JOINT, "width": 0.8},
                    opacity=0.55,
                    name="joint NUTS, one line per chain",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="retained draw", row=1, col=col)
    fig.update_yaxes(title_text="log σ", row=1, col=1)
    fig.update_annotations(font={"size": 12, "color": C_INK})
    return _dedupe_legend(
        _layout(fig, title="Real joint NUTS fits: four chains each, log σ", height=420)
    )


def energy_check(run: FitResult) -> tuple[float, float, float]:
    """NUTS's energy check: the SD of the energy across draws, of its change per draw, and E-BFMI."""
    energy = np.asarray(run.extra["energy"], dtype=float)
    change = np.diff(energy, axis=1)
    bfmi = np.mean(
        np.sum(change**2, axis=1)
        / np.sum((energy - energy.mean(axis=1, keepdims=True)) ** 2, axis=1)
    )
    return float(energy.std(axis=1).mean()), float(change.std(axis=1).mean()), float(bfmi)


def fig_energy(runs: Mapping[str, FitResult]) -> go.Figure:
    """Per chart: how far the energy spreads across draws, against how far one draw moves it."""
    names = {"centred": "centred chart", "non_centred": "innovation chart"}
    fig = make_subplots(
        rows=1,
        cols=len(runs),
        subplot_titles=[
            f"{names[chart]}: E-BFMI {energy_check(run)[2]:.2f}" for chart, run in runs.items()
        ],
        horizontal_spacing=0.08,
    )
    for col, run in enumerate(runs.values(), start=1):
        energy = np.asarray(run.extra["energy"], dtype=float)
        for values, colour, name in (
            (
                (energy - energy.mean(axis=1, keepdims=True)).ravel(),
                C_MUTED,
                "energy across draws, about its mean",
            ),
            (
                np.diff(energy, axis=1).ravel(),
                C_JOINT,
                "change in energy from one draw to the next",
            ),
        ):
            fig.add_trace(
                go.Histogram(
                    x=values,
                    xbins={"start": -150, "end": 150, "size": 5},
                    histnorm="probability density",
                    marker={"color": colour},
                    opacity=0.7,
                    name=name,
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="energy", range=[-150, 150], row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)
    fig.update_layout(barmode="overlay")
    fig.update_annotations(font={"size": 12, "color": C_INK})
    return _dedupe_legend(
        _layout(
            fig,
            title="NUTS's own check: one draw's energy change against the spread it must cover (real fits)",
            height=380,
        )
    )


def fig_votes(scale: Votes, effect: Votes) -> go.Figure:
    """Chance that θ′ wins, judged by today's path or with the path integrated out (exact)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["move σ (coupled)", "move β1 (loose)"],
        horizontal_spacing=0.08,
    )
    for col, votes_ in enumerate((scale, effect), start=1):
        fig.add_trace(
            go.Scatter(
                x=votes_.distances,
                y=votes_.integrated,
                mode="lines",
                line={"color": C_INK_2, "width": 2, "dash": "dash"},
                name="path integrated out: what the label move aims for",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=votes_.distances,
                y=votes_.one_path,
                mode="lines+markers",
                line={"color": C_LABEL, "width": 2},
                marker={"size": 8},
                name="judged by today's path alone",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="distance of θ′ from θ (posterior SDs)", row=1, col=col)
        fig.update_yaxes(range=[0, 0.55], row=1, col=col)
    fig.update_yaxes(title_text="chance θ′ wins", row=1, col=1)
    return _dedupe_legend(_layout(fig, title="Who decides between θ and θ′ (exact)", height=380))


def fig_label_odds(run: FitResult) -> go.Figure:
    """The real label move: each sweep's chance of moving to θ′, and where θ ended up."""
    odds = np.exp(run.extra["final_label_log_probs"][..., 1]).ravel()
    moved = run.extra["selected_parameter_label"].ravel()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["chance that θ′ won, per sweep", "label kept or moved"],
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.1,
    )
    fig.add_trace(
        go.Histogram(
            x=odds,
            nbinsx=40,
            marker={"color": C_LABEL, "line": {"color": "white", "width": 1}},
            name="sweeps",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["kept θ", "moved to θ′"],
            y=[float(np.mean(moved == 0)), float(np.mean(moved == 1))],
            marker={"color": [C_MUTED, C_LABEL]},
            text=[f"{np.mean(moved == 0):.0%}", f"{np.mean(moved == 1):.0%}"],
            textposition="outside",
            name="share of sweeps",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="P(θ′ | new path)", row=1, col=1)
    fig.update_yaxes(title_text="sweeps", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.1], row=1, col=2)
    return _layout(
        fig,
        title="Inside the real label move (MPGibbs alone, 4,000 sweeps)",
        height=360,
        legend=False,
    )


def fig_plane(
    exact: Exact, runs: Mapping[str, tuple[FitResult, str]], *, sweeps: int = 80
) -> go.Figure:
    """The first retained sweeps of one chain per sampler, over the exact posterior of (β1, log σ)."""
    centre, spread = float(exact.effect_means[0]), float(exact.effect_sd[0])
    values = np.linspace(centre - 4 * spread, centre + 4 * spread, 81)
    density = exact.plane(values)[::4]
    fig = make_subplots(rows=1, cols=len(runs), subplot_titles=list(runs), horizontal_spacing=0.08)
    for col, (label, (run, colour)) in enumerate(runs.items(), start=1):
        fig.add_trace(
            go.Contour(
                x=values,
                y=exact.log_scales[::4],
                z=density,
                showscale=False,
                contours={"coloring": "lines", "showlabels": False},
                line={"width": 1},
                colorscale=[[0, C_GRID], [1, C_INK_2]],
                ncontours=8,
                name="exact posterior",
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=run.column("beta1")[0, :sweeps],
                y=run.column("sigma")[0, :sweeps],
                mode="lines+markers",
                line={"color": colour, "width": 1.2},
                marker={"size": 6, "color": colour},
                name=label,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(range=[values[0], values[-1]], title_text="β1", row=1, col=col)
        fig.update_yaxes(
            range=[exact.scale_mean - 4 * exact.scale_sd, exact.scale_mean + 4 * exact.scale_sd],
            title_text="log σ",
            row=1,
            col=col,
        )
    return _layout(
        fig,
        title=f"The first {sweeps} retained draws of one chain, over the exact posterior",
        height=430,
        legend=False,
    )


def fig_smoother_refresh(history: History, runs: Mapping[str, FitResult]) -> go.Figure:
    """How far each point of the path moves per sweep, and how often a sweep keeps today's path."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "path movement per sweep, by time",
            "share of sweeps keeping today's value",
        ],
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.1,
    )
    colours = [C_SMOOTH, C_MUTED]
    for index, (label, run) in enumerate(runs.items()):
        movement = run.extra["latent_move_rms_per_t"].mean(axis=(0, 1))
        fig.add_trace(
            go.Scatter(
                x=history.times,
                y=movement,
                mode="lines",
                line={"color": colours[index], "width": 2},
                name=label,
            ),
            row=1,
            col=1,
        )
        kept = float(run.extra["reference_path_hit_rate"].mean())
        fig.add_trace(
            go.Bar(
                x=[label.split(" (")[0].split(",")[0]],
                y=[kept],
                marker={"color": colours[index]},
                text=[f"{kept:.0%}"],
                textposition="outside",
                name=label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="day", row=1, col=1)
    fig.update_yaxes(title_text="RMS move of x(t)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], row=1, col=2)
    return _layout(fig, title="How the real smoothers renew the path", height=380)


def fig_arm_traces(
    runs: Mapping[str, tuple[FitResult, str]], exact: Exact, *, draws: int = 400
) -> go.Figure:
    """One chain per sampler: β1 and log σ against the exact ±2 SD."""
    columns = (("β1: loose", "beta1"), ("log σ: coupled", "sigma"))
    titles = []
    for label, (run, _) in runs.items():
        for heading, name in columns:
            values = run.column(name)
            titles.append(
                f"{label} · {heading} · {effective_per_thousand(values, run.ess[name]):.0f} effective per 1,000"
            )
    fig = make_subplots(
        rows=len(runs),
        cols=2,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.07,
    )
    bands = {
        "beta1": (float(exact.effect_means[0]), float(exact.effect_sd[0])),
        "sigma": (exact.scale_mean, exact.scale_sd),
    }
    for row, (label, (run, colour)) in enumerate(runs.items(), start=1):
        for col, (_, name) in enumerate(columns, start=1):
            mean, sd = bands[name]
            xs = np.arange(draws)
            _band(
                fig,
                xs,
                mean - 2 * sd,
                mean + 2 * sd,
                row=row,
                col=col,
                name="exact posterior ±2 SD",
            )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=run.column(name)[0, :draws],
                    mode="lines",
                    line={"color": colour, "width": 1.2},
                    name=label,
                ),
                row=row,
                col=col,
            )
    for col in (1, 2):
        fig.update_xaxes(title_text="retained draw", row=len(runs), col=col)
    fig.update_annotations(font={"size": 12, "color": C_INK})
    return _dedupe_legend(
        _layout(
            fig,
            title=f"The first {draws} retained draws of one chain (real fits)",
            height=250 * len(runs) + 120,
        )
    )


def fig_autocorrelation(runs: Mapping[str, tuple[FitResult, str]], *, lags: int = 60) -> go.Figure:
    """How fast each sampler forgets where it was, for a loose and a coupled parameter."""
    columns = (("β1: loose", "beta1"), ("log σ: coupled", "sigma"))
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[heading for heading, _ in columns], horizontal_spacing=0.08
    )
    for col, (_, name) in enumerate(columns, start=1):
        for label, (run, colour) in runs.items():
            curve = autocorrelation(run.column(name), lags)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(lags + 1),
                    y=curve,
                    mode="lines",
                    line={"color": colour, "width": 2},
                    name=label,
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="lag (draws)", row=1, col=col)
        fig.update_yaxes(range=[-0.15, 1.02], row=1, col=col)
    fig.update_yaxes(title_text="autocorrelation", row=1, col=1)
    return _dedupe_legend(
        _layout(fig, title="How fast each sampler forgets (real fits)", height=380)
    )


def summary_rows(
    runs: Mapping[str, FitResult], *, gradients: Mapping[str, str]
) -> list[dict[str, str]]:
    """Effective draws per 1,000, R̂ and gradient work per draw, from each fit's own diagnostics."""
    rows = []
    for label, run in runs.items():
        effects = [
            effective_per_thousand(run.column(f"beta{j + 1}"), run.ess[f"beta{j + 1}"])
            for j in range(EFFECTS)
        ]
        work = gradients[label]
        rows.append(
            {
                "sampler": label,
                "effects: effective per 1,000 (worst)": f"{min(effects):.0f}",
                "log σ: effective per 1,000": f"{effective_per_thousand(run.column('sigma'), run.ess['sigma']):.0f}",
                "path: effective per 1,000 (worst point)": f"{1000 * run.path_ess / run.draws[..., 0].size:.0f}",
                "worst R̂": f"{max(run.r_hat.values()):.3f}",
                "gradients per draw (median)": f"{np.median(run.extra[work]):.0f}"
                if work
                else "none",
                "native check": run.verdict,
            }
        )
    return rows
