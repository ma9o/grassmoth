"""Tests for emission log-probability functions.

Covers: Gaussian, Poisson, Student-t, Gamma, Bernoulli, NegBin, Beta,
        probit-link variants, inverse-link Gamma, and get_emission_fn dispatcher.
"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import pytest

from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.execution.emissions import gaussian_block_log_prob
from tests.observation_fixtures import mean_density, observation_kernel, observation_laws

# =============================================================================
# Helpers
# =============================================================================


def _simple_params(n_latent=2, n_manifest=2):
    """Identity measurement structure: H=I, d=0, R=0.5*I."""
    H = jnp.eye(n_manifest, n_latent)
    d = jnp.zeros(n_manifest)
    R = jnp.eye(n_manifest) * 0.5
    return H, d, R


# =============================================================================
# Gaussian
# =============================================================================


@pytest.mark.inference(concern="sampling")
class TestGaussianEmission:
    @pytest.mark.parametrize("mask", [[1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 1]])
    def test_exact_marginal_and_gradients_ignore_missing_covariance(self, mask):
        """Missing variances can be arbitrarily large without changing the observed law."""
        mask = jnp.array(mask, dtype=bool)
        covariance = jnp.array([[2.0, 1e5, 0.3], [1e5, 1e12, 2e5], [0.3, 2e5, 3.0]])
        mean = jnp.array([0.5, -2.0, 0.2])
        values = jnp.array([1.0, 3.0, -1.0])
        observed = jnp.where(mask)[0]

        def reference(location):
            if not len(observed):
                return jnp.array(0.0)
            return jstats.multivariate_normal.logpdf(
                values[observed], location[observed], covariance[jnp.ix_(observed, observed)]
            )

        def score(location):
            return gaussian_block_log_prob(
                jnp.where(mask, values, jnp.nan), location, covariance, mask
            )

        assert float(score(mean)) == pytest.approx(float(reference(mean)), rel=2e-7, abs=1e-6)
        assert jnp.allclose(jax.grad(score)(mean), jax.grad(reference)(mean), atol=1e-6)

    def test_missing_channel_ignored(self):
        """Masked-out channels should not affect log-prob."""
        H, d, R = _simple_params()
        z = jnp.array([1.0, 2.0])
        y_close = jnp.array([1.0, 2.0])
        y_far = jnp.array([1.0, 999.0])
        mask = jnp.array([1.0, 0.0])
        lp_close = gaussian_block_log_prob(y_close, H @ z + d, R, mask)
        lp_far = gaussian_block_log_prob(y_far, H @ z + d, R, mask)
        assert jnp.isclose(lp_close, lp_far, atol=1e-3)

    def test_all_missing_returns_zero(self):
        """When all channels are missing, log-prob should be 0."""
        H, d, R = _simple_params()
        z = jnp.array([1.0, 2.0])
        y = jnp.array([999.0, 999.0])
        mask = jnp.zeros(2)
        lp = gaussian_block_log_prob(y, H @ z + d, R, mask)
        assert jnp.isclose(lp, 0.0)


# =============================================================================
# Poisson
# =============================================================================


@pytest.mark.inference(concern="sampling")
class TestPoissonEmission:
    def test_matches_scipy(self):
        """Log-prob should match jax.scipy.stats.poisson.logpmf."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([jnp.log(5.0)])
        y = jnp.array([3.0])
        mask = jnp.ones(1)
        lp = observation_kernel(
            [DistributionFamily.POISSON] * (H @ z + d).shape[-1],
            [LinkFunction.LOG] * (H @ z + d).shape[-1],
            None,
        ).log_prob_fn(y, H @ z + d, R, mask)
        expected = jstats.poisson.logpmf(3.0, 5.0)
        assert jnp.isclose(lp, expected, atol=1e-5)

    def test_masked_channel_zero(self):
        """Masked channels contribute 0 to log-prob."""
        H = jnp.eye(2)
        d = jnp.zeros(2)
        R = jnp.eye(2)
        z = jnp.array([jnp.log(5.0), jnp.log(10.0)])
        y = jnp.array([3.0, 999.0])
        mask = jnp.array([1.0, 0.0])
        lp = observation_kernel(
            [DistributionFamily.POISSON] * (H @ z + d).shape[-1],
            [LinkFunction.LOG] * (H @ z + d).shape[-1],
            None,
        ).log_prob_fn(y, H @ z + d, R, mask)
        expected = jstats.poisson.logpmf(3.0, 5.0)
        assert jnp.isclose(lp, expected, atol=1e-5)


# =============================================================================
# Student-t
# =============================================================================


@pytest.mark.inference(concern="sampling")
class TestStudentTEmission:
    def test_heavier_tails_than_gaussian(self):
        """Student-t should give higher log-prob for outliers than Gaussian."""
        H, d, R = _simple_params(1, 1)
        z = jnp.array([0.0])
        y = jnp.array([5.0])
        mask = jnp.ones(1)
        lp_t = observation_kernel(
            [DistributionFamily.STUDENT_T] * (H @ z + d).shape[-1],
            [LinkFunction.IDENTITY] * (H @ z + d).shape[-1],
            {"obs_df": 3.0},
        ).log_prob_fn(y, H @ z + d, R, mask)
        lp_g = gaussian_block_log_prob(y, H @ z + d, R, mask)
        assert lp_t > lp_g

    def test_matches_scipy_univariate(self):
        """Should match scipy t.logpdf for scalar case."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1) * 2.0
        z = jnp.array([1.0])
        y = jnp.array([3.0])
        mask = jnp.ones(1)
        df = 5.0
        lp = observation_kernel(
            [DistributionFamily.STUDENT_T] * (H @ z + d).shape[-1],
            [LinkFunction.IDENTITY] * (H @ z + d).shape[-1],
            {"obs_df": df},
        ).log_prob_fn(y, H @ z + d, R, mask)
        scale = jnp.sqrt(2.0)
        expected = jstats.t.logpdf(3.0, df, loc=1.0, scale=scale)
        assert jnp.isclose(lp, expected, atol=1e-5)


# =============================================================================
# Gamma
# =============================================================================


@pytest.mark.inference(concern="sampling")
class TestGammaEmission:
    def test_inverse_link(self):
        """Gamma with inverse link: mean = 1/eta, scale = mean/shape."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.5])
        y = jnp.array([1.5])
        mask = jnp.ones(1)
        lp = observation_kernel(
            [DistributionFamily.GAMMA] * (H @ z + d).shape[-1],
            [LinkFunction.INVERSE] * (H @ z + d).shape[-1],
            {"obs_shape": 2.0},
        ).log_prob_fn(y, H @ z + d, R, mask)
        # mean = 1/0.5 = 2.0, scale = 2.0/2.0 = 1.0
        expected = jstats.gamma.logpdf(1.5, a=2.0, scale=1.0)
        assert jnp.isclose(lp, expected, atol=1e-5)

    def test_log_link_invalid_observation_returns_negative_infinity(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([jnp.log(2.0)])
        y = jnp.array([0.0])
        mask = jnp.ones(1)
        fn = observation_kernel([DistributionFamily.GAMMA], None, {"obs_shape": 2.0}).log_prob_fn
        lp = fn(y, H @ z + d, R, mask)
        assert jnp.isneginf(lp)

    def test_inverse_link_invalid_linear_predictor_returns_negative_infinity(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([-0.5])
        y = jnp.array([1.5])
        mask = jnp.ones(1)
        lp = observation_kernel(
            [DistributionFamily.GAMMA] * (H @ z + d).shape[-1],
            [LinkFunction.INVERSE] * (H @ z + d).shape[-1],
            {"obs_shape": 2.0},
        ).log_prob_fn(y, H @ z + d, R, mask)
        assert jnp.isneginf(lp)


# =============================================================================
# Bernoulli
# =============================================================================


@pytest.mark.inference(concern="sampling")
class TestBernoulliEmission:
    def test_probit_link(self):
        """Probit link should give log(0.5) at eta=0."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([1.0])
        mask = jnp.ones(1)
        lp_logit = observation_kernel(
            [DistributionFamily.BERNOULLI] * (H @ z + d).shape[-1],
            [LinkFunction.LOGIT] * (H @ z + d).shape[-1],
            None,
        ).log_prob_fn(y, H @ z + d, R, mask)
        lp_probit = observation_kernel(
            [DistributionFamily.BERNOULLI] * (H @ z + d).shape[-1],
            [LinkFunction.PROBIT] * (H @ z + d).shape[-1],
            None,
        ).log_prob_fn(y, H @ z + d, R, mask)
        assert jnp.isclose(lp_logit, jnp.log(0.5), atol=1e-5)
        assert jnp.isclose(lp_probit, jnp.log(0.5), atol=1e-5)


# =============================================================================
# Negative Binomial
# =============================================================================


@pytest.mark.inference(concern="sampling")
class TestNegBinEmission:
    def test_overdispersion_increases_with_lower_r(self):
        """Lower r means more overdispersion, so NB should be more spread."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([jnp.log(5.0)])
        y = jnp.array([3.0])
        mask = jnp.ones(1)
        lp_low_r = observation_kernel(
            [DistributionFamily.NEGATIVE_BINOMIAL] * (H @ z + d).shape[-1],
            [LinkFunction.LOG] * (H @ z + d).shape[-1],
            {"obs_r": 2.0},
        ).log_prob_fn(y, H @ z + d, R, mask)
        lp_high_r = observation_kernel(
            [DistributionFamily.NEGATIVE_BINOMIAL] * (H @ z + d).shape[-1],
            [LinkFunction.LOG] * (H @ z + d).shape[-1],
            {"obs_r": 100.0},
        ).log_prob_fn(y, H @ z + d, R, mask)
        # Higher r (less overdispersion) should give higher log-prob near the mean
        assert lp_high_r > lp_low_r


# =============================================================================
# Ordered Logistic / Categorical
# =============================================================================


@pytest.mark.inference(concern="sampling")
class TestDiscreteEmission:
    def test_ordered_logistic_matches_manual_probability(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([1.0])
        mask = jnp.ones(1)
        cutpoints = jnp.array([[-1.0, 1.0]])
        level_counts = jnp.array([3])

        lp = observation_kernel(
            [DistributionFamily.ORDERED_LOGISTIC] * (H @ z + d).shape[-1],
            parameters={"obs_ordered_cutpoints": cutpoints, "obs_level_counts": level_counts},
        ).log_prob_fn(y, H @ z + d, R, mask)
        expected = jnp.log(jax.nn.sigmoid(1.0) - jax.nn.sigmoid(-1.0))
        assert jnp.isclose(lp, expected, atol=1e-5)

    def test_categorical_matches_manual_softmax(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.7])
        y = jnp.array([2.0])
        mask = jnp.ones(1)
        intercepts = jnp.array([[-1.0, 0.5]])
        slopes = jnp.array([[0.2, -0.4]])
        level_counts = jnp.array([3])

        lp = observation_kernel(
            [DistributionFamily.CATEGORICAL] * (H @ z + d).shape[-1],
            parameters={
                "obs_cat_intercepts": intercepts,
                "obs_cat_slopes": slopes,
                "obs_level_counts": level_counts,
            },
        ).log_prob_fn(y, H @ z + d, R, mask)
        logits = jnp.array([0.0, -1.0 + 0.2 * 0.7, 0.5 - 0.4 * 0.7])
        expected = jax.nn.log_softmax(logits)[2]
        assert jnp.isclose(lp, expected, atol=1e-5)


# =============================================================================
# Beta
# =============================================================================


@pytest.mark.inference(concern="sampling")
class TestBetaEmission:
    def test_probit_link_at_center(self):
        """Beta probit at eta=0: Phi(0)=0.5 → Beta(0.5|5,5), must be a valid positive density."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([0.5])
        mask = jnp.ones(1)
        lp = observation_kernel(
            [DistributionFamily.BETA] * (H @ z + d).shape[-1],
            [LinkFunction.PROBIT] * (H @ z + d).shape[-1],
            {"obs_concentration": 10.0},
        ).log_prob_fn(y, H @ z + d, R, mask)
        # Phi(0)=0.5, concentration=10 → alpha=beta=5, y=0.5 is mode → high density
        assert lp > 0.0, f"Log-prob at mode of symmetric Beta should be positive, got {lp}"

    def test_logit_vs_probit_at_center(self):
        """At eta=0, logit and probit both give mean=0.5."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([0.5])
        mask = jnp.ones(1)
        conc = 10.0
        lp_logit = observation_kernel(
            [DistributionFamily.BETA] * (H @ z + d).shape[-1],
            [LinkFunction.LOGIT] * (H @ z + d).shape[-1],
            {"obs_concentration": conc},
        ).log_prob_fn(y, H @ z + d, R, mask)
        lp_probit = observation_kernel(
            [DistributionFamily.BETA] * (H @ z + d).shape[-1],
            [LinkFunction.PROBIT] * (H @ z + d).shape[-1],
            {"obs_concentration": conc},
        ).log_prob_fn(y, H @ z + d, R, mask)
        assert jnp.isclose(lp_logit, lp_probit, atol=1e-4)

    def test_invalid_observation_returns_negative_infinity(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([1.0])
        mask = jnp.ones(1)
        lp = observation_kernel(
            [DistributionFamily.BETA] * (H @ z + d).shape[-1],
            [LinkFunction.LOGIT] * (H @ z + d).shape[-1],
            {"obs_concentration": 10.0},
        ).log_prob_fn(y, H @ z + d, R, mask)
        assert jnp.isneginf(lp)


@pytest.mark.inference(concern="sampling")
class TestMeanParamLogProb:
    def test_gamma_invalid_support_returns_negative_infinity(self):
        fn = mean_density(
            observation_laws([DistributionFamily.GAMMA], parameters={"obs_shape": 2.0})[0]
        )
        lp = fn(
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([2.0], dtype=jnp.float32),
            jnp.eye(1, dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )
        assert jnp.isneginf(lp)

    def test_beta_invalid_mean_returns_negative_infinity(self):
        fn = mean_density(
            observation_laws([DistributionFamily.BETA], parameters={"obs_concentration": 10.0})[0]
        )
        lp = fn(
            jnp.array([0.5], dtype=jnp.float32),
            jnp.array([1.2], dtype=jnp.float32),
            jnp.eye(1, dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )
        assert jnp.isneginf(lp)
