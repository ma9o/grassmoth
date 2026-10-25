"""Browser prior plots preserve the densities of serialized native scalar laws."""

import json
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy import stats

from grassmoth.artifacts.probability.families import PriorDistributionFamily
from grassmoth.artifacts.probability.serialization import encode_distribution

pytestmark = pytest.mark.inference(concern="sampling")


def test_browser_authored_plots_match_native_laws(tmp_path):
    """Compare every authored family and truncation direction against its NumPyro law."""
    authored_laws = {
        "Normal": dist.Normal(jnp.array(-1.5), jnp.array(2.0)),
        "HalfNormal": dist.HalfNormal(0.7),
        "Beta": dist.Beta(2.0, 5.0),
        "Uniform": dist.Uniform(-3.0, 7.0),
        "TruncatedNormal": dist.TruncatedNormal(0.5, 1.2, low=-1.0, high=2.0),
        "Gamma": dist.Gamma(2.3, 4.0),
        "LogNormal": dist.LogNormal(-0.2, 0.6),
        "Exponential": dist.Exponential(2.5),
    }
    assert set(authored_laws) == {family.value for family in PriorDistributionFamily}
    laws = (
        *authored_laws.values(),
        dist.Beta(0.4, 0.7),
        dist.TruncatedNormal(0.5, 1.2, low=1.5),
        dist.TruncatedNormal(0.0, 1.0, low=jnp.array(0.0), high=jnp.array(np.inf)),
        dist.TruncatedNormal(0.5, 1.2, high=-1.0),
    )
    far_tail = dist.TruncatedNormal(0.0, 1.0, low=5.0, high=5.2)
    source = (
        Path(__file__).resolve().parents[4]
        / "apps/grassmoth-client/src/lib/model-asset/authored-prior-plot.ts"
    )
    runner = tmp_path / "prior-plots.ts"
    runner.write_text(
        f"import {{ authoredPriorPlot }} from {json.dumps(str(source))};\n"
        "let input = '';\n"
        "for await (const chunk of process.stdin) input += chunk;\n"
        "process.stdout.write(JSON.stringify(JSON.parse(input).map(authoredPriorPlot)));\n"
    )
    result = subprocess.run(
        ["bun", str(runner)],
        input=json.dumps([encode_distribution(law) for law in (*laws, far_tail)]),
        capture_output=True,
        text=True,
        check=True,
    )
    *curves, far_tail_curve = json.loads(result.stdout)
    for law, curve in zip(laws, curves, strict=True):
        x = np.asarray(curve["x"])
        _assert_plot(curve, np.exp(law.log_prob(x)), law.cdf(x[[0, -1]]))
    # The browser plots in doubles. Five scales into the tail, NumPyro's float32
    # normalizer cancels, so SciPy checks the browser's reflected tail instead.
    x = np.asarray(far_tail_curve["x"])
    reference = stats.truncnorm(5.0, 5.2)
    _assert_plot(far_tail_curve, reference.pdf(x), reference.cdf(x[[0, -1]]))


def _assert_plot(curve, density, ends):
    assert len(curve["x"]) == 129
    assert np.all(np.diff(curve["x"]) > 0)
    np.testing.assert_allclose(curve["density"], density, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(ends, [0.01, 0.99], atol=2e-5)
