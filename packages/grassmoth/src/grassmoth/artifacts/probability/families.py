"""Scientific probability-family identities shared by laws and authoring."""

from enum import StrEnum


class DistributionFamily(StrEnum):
    """This enumeration identifies the probability family used to model an observed variable."""

    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    POISSON = "poisson"
    GAMMA = "gamma"
    BERNOULLI = "bernoulli"
    NEGATIVE_BINOMIAL = "negative_binomial"
    BETA = "beta"
    ORDERED_LOGISTIC = "ordered_logistic"
    CATEGORICAL = "categorical"
    DELTA = "delta"

    @property
    def is_discrete(self) -> bool:
        """Whether this family has discrete (integer) support."""
        return self in {
            DistributionFamily.BERNOULLI,
            DistributionFamily.POISSON,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
        }

    @property
    def support_interior_point(self) -> float:
        """A scalar strictly inside this family's support (for dummy observations)."""
        if self == DistributionFamily.GAMMA:
            return 1.0
        if self == DistributionFamily.BETA:
            return 0.5
        return 0.0

    @property
    def uses_manifest_noise(self) -> bool:
        """Whether this family's emission log-prob reads per-channel manifest noise R.

        Only Gaussian and Student-t emissions read R in the ``emission_log_prob_*``
        functions and the posterior-predictive switch branches. All other
        families determine observation variance from family-level hyperparameters
        (``obs_r``, ``obs_shape``, ``obs_concentration``, ...) and mark R as
        unused with the ``_R`` / ``_std`` naming convention. Emitting a free
        ``obs_sd_<indicator>`` parameter for a non-{Gaussian, Student-t} channel
        therefore creates a disconnected parameter that contributes nothing to
        the likelihood.
        """
        return self in {DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T}


class PriorDistributionFamily(StrEnum):
    """Probability families permitted in authored prior proposals."""

    NORMAL = "Normal"
    HALF_NORMAL = "HalfNormal"
    BETA = "Beta"
    UNIFORM = "Uniform"
    TRUNCATED_NORMAL = "TruncatedNormal"
    GAMMA = "Gamma"
    LOG_NORMAL = "LogNormal"
    EXPONENTIAL = "Exponential"
