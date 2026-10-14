"""HTTP response adapter for the shared numerical MessagePack codec."""

from typing import override

from fastapi.responses import JSONResponse

from nof1_causal_lab.study.persistence.codec import pack_payload


class MessagePackResponse(JSONResponse):
    """Preserve FastAPI's operation schema while serving the numerical wire codec."""

    media_type = "application/msgpack"

    @override
    def render(self, content: object) -> bytes:
        """Encode metadata already normalized by the response owner."""
        return pack_payload(content)
