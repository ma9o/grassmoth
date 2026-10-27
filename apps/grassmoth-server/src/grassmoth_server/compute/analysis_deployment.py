"""Daily collection of analysis-cache entries older than fourteen days."""

import modal

from grassmoth_server.compute.analysis_cache import (
    ANALYSIS_CACHE_ROOT,
    ANALYSIS_CACHE_VOLUME,
    cleanup_analysis_cache,
)
from grassmoth_server.compute.modal_images import cpu_image

app = modal.App("grassmoth-fit-analysis-cache", image=cpu_image)
app.function(
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=300,
    retries=0,
    schedule=modal.Period(days=1),
    env={"JAX_PLATFORMS": "cpu"},
    volumes={
        str(ANALYSIS_CACHE_ROOT): modal.Volume.from_name(
            ANALYSIS_CACHE_VOLUME, create_if_missing=True
        ),
    },
)(cleanup_analysis_cache)
