"""HTTP response adapter for the shared numerical MessagePack codec."""

import re
from typing import override

from fastapi.responses import JSONResponse

from grassmoth_server.study.persistence.codec import pack_payload


def prefers_json(accept: str) -> bool:
    """Compare Accept qualities, using specific ranges before wildcards and binary ties."""
    ranges: dict[str, float] = {}
    for item in accept.lower().split(","):
        media_type, *parameters = item.strip().split(";")
        quality = 1.0
        for parameter in parameters:
            name, separator, value = parameter.strip().partition("=")
            if name == "q" and separator:
                quality = (
                    float(value)
                    if re.fullmatch(r"(?:0(?:\.\d{0,3})?|1(?:\.0{0,3})?)", value)
                    else 0.0
                )
        ranges[media_type.strip()] = max(ranges.get(media_type.strip(), 0.0), quality)

    def effective_quality(media_type: str) -> float:
        return next(
            (ranges[key] for key in (media_type, "application/*", "*/*") if key in ranges),
            0.0,
        )

    return effective_quality("application/json") > effective_quality("application/msgpack")


class MessagePackResponse(JSONResponse):
    """Preserve FastAPI's operation schema while serving the numerical wire codec."""

    media_type = "application/msgpack"

    @override
    def render(self, content: object) -> bytes:
        """Encode metadata already normalized by the response owner."""
        return pack_payload(content)
