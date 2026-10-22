"""Modal deployment of saved-study HTTP reads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import modal

from grassmoth_server.compute.modal_images import cpu_image

if TYPE_CHECKING:
    from fastapi import FastAPI


app = modal.App("grassmoth-server", image=cpu_image)
secrets = modal.Secret.from_name("grassmoth-server-secrets")


@app.function(
    image=cpu_image,
    env={"READ_ONLY_FACADE": "1"},
    secrets=[secrets],
)
@modal.asgi_app()
def read_facade() -> FastAPI:
    """The hosted viewer's backend: journal reads over the R2 store, no moves."""
    from grassmoth_server.api.read_only import create_read_facade_app

    return create_read_facade_app()
