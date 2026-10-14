"""Shared parameter and sample-site vocabulary."""

from enum import Enum, StrEnum
from typing import Annotated

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value


class SupportClass(Enum):
    """Runtime support class for a sample site."""

    REAL = "real"
    POSITIVE = "positive"
    CORRELATION = "correlation"


class SiteKind(Enum):
    """Semantic role for each sample site."""

    DYNAMICS_DECAY = "dynamics_decay"
    DYNAMICS_CINT = "dynamics_cint"
    DYNAMICS_WEIGHT = "dynamics_weight"
    DYNAMICS_POTENTIAL_CENTER = "dynamics_potential_center"
    DYNAMICS_POTENTIAL_QUARTIC = "dynamics_potential_quartic"
    HILL_EMAX = "hill_emax"
    HILL_EC50 = "hill_ec50"
    HILL_N = "hill_n"
    DIFFUSION_DIAG = "diffusion_diag"
    DIFFUSION_LOWER = "diffusion_lower"
    STATIC_STATE_SD = "static_state_sd"
    LOADING = "loading"
    MANIFEST_MEANS = "manifest_means"
    MANIFEST_VAR_DIAG = "manifest_var_diag"
    T0_MEANS = "t0_means"
    T0_VAR_DIAG = "t0_var_diag"
    T0_VAR_LOWER = "t0_var_lower"
    OBS_DF = "obs_df"
    OBS_SHAPE = "obs_shape"
    OBS_R = "obs_r"
    OBS_CONCENTRATION = "obs_concentration"
    OBS_ORDERED_BASE = "obs_ordered_base"
    OBS_ORDERED_GAPS = "obs_ordered_gaps"
    OBS_CAT_INTERCEPTS = "obs_cat_intercepts"
    OBS_CAT_SLOPES = "obs_cat_slopes"
    PROC_DF = "proc_df"


class PriorAuthoringTransform(StrEnum):
    """How an authored semantic prior is transformed before site attachment."""

    IDENTITY = "identity"
    DT_PERSISTENCE_TO_CT_DECAY = "dt_persistence_to_ct_decay"
    DT_EFFECT_TO_CT_RATE = "dt_effect_to_ct_rate"
    INITIAL_STATE_CORRELATION = "initial_state_correlation"


class ParameterCoordinate(Value):
    """A parameter coordinate identifies a scalar element of a named runtime sample site."""

    site_name: str
    indices: tuple[Annotated[int, Field(ge=0)], ...]

    @property
    def label(self) -> str:
        """Human-readable coordinate label; identity remains the explicit site and indices."""
        suffix = "[" + ",".join(str(index) for index in self.indices) + "]" if self.indices else ""
        return self.site_name + suffix
