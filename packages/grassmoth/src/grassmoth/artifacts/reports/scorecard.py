"""Optional retained-draw movement summaries; never production convergence gates."""

from pydantic import Field, FiniteFloat

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import ConstructRef


class FrequencyRefreshBand(Value):
    """Median squared adjacent-draw jump divided by twice pooled mode variance."""

    start_mode: int = Field(ge=0, description="Inclusive orthonormal DCT-II mode index.")
    stop_mode: int = Field(gt=0, description="Exclusive orthonormal DCT-II mode index.")
    median_refresh: FiniteFloat | None = Field(
        description="Null without two retained draws per chain; zero for constant modes."
    )


class FrequencyRefresh(Value):
    """Frequency-localized movement for one construct's complete retained trajectory."""

    subject: ConstructRef
    label: str
    bands: tuple[FrequencyRefreshBand, ...]


class MovementQuantiles(Value):
    """Minimum, median and maximum across selected time coordinates or native delta values."""

    minimum: FiniteFloat
    median: FiniteFloat
    maximum: FiniteFloat


class MovementWindow(Value):
    """An existing question's half-open window expressed in retained model days."""

    name: str
    start_day: FiniteFloat
    end_day: FiniteFloat


class MovementRegion(Value):
    """Refresh and final native delta on independent coordinates in one window.

    For adapted kappa=1, delta is the frozen scalar multiplier of the coordinate
    geometry; these are not the effective per-coordinate step sizes.
    """

    window: MovementWindow
    coordinates: int = Field(ge=0)
    refresh_quantiles: MovementQuantiles | None
    delta_quantiles: MovementQuantiles | None


class ReadingMovement(Value):
    """Pooled movement at free grid coordinates with or without any prepared reading."""

    has_readings: bool
    coordinates: int = Field(ge=0)
    stretches: int = Field(ge=0, description="Contiguous stretches on the retained time grid.")
    refresh_quantiles: MovementQuantiles | None
    delta_quantiles: MovementQuantiles | None


class WorstTimeMovement(Value):
    """An independent time coordinate ordered by its retained-draw refresh ratio."""

    time_index: int = Field(ge=0)
    time_day: FiniteFloat
    refresh: FiniteFloat


class RegionalMovement(Value):
    """Declared windows and the slowest independent time coordinates of one construct."""

    subject: ConstructRef
    label: str
    regions: tuple[MovementRegion, ...]
    reading_coverage: tuple[ReadingMovement, ...]
    worst_times: tuple[WorstTimeMovement, ...]
