"""Read a receipt photo: one call through the AcruxCore gateway.

The prompt lives in AcruxCore as `receipt-extract`; the alias picks the version.
The gateway records a trace linked to that prompt version.
"""
import base64
import json
import mimetypes
import re
import time
from dataclasses import dataclass

import acruxcore as acrux

from . import config


@dataclass
class Fields:
    company: str
    date: str
    total: str
    trace_id: str | None
    raw: str
    seconds: float


def parse_fields(reply: str) -> dict[str, str]:
    """Pulls the JSON object out of a model reply, tolerating a code fence or stray prose.

    Raises:
        ValueError: the reply holds no JSON object.
    """
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        raise ValueError(f"no JSON object in reply: {reply[:120]!r}")
    data = json.loads(match.group(0))
    return {k: str(data.get(k, "")).strip() for k in ("company", "date", "total")}


def to_data_url(content: bytes, filename: str) -> str:
    """Encodes an uploaded photo as a ``data:`` URL, the form the gateway accepts for a file."""
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(content).decode()}"


async def extract_fields(receipt_url: str, model: str | None = None, alias: str | None = None, tags: list[str] | None = None) -> Fields:
    """Renders the prompt for ``alias`` with the image and runs it on a registered vision model.

    Args:
        receipt_url: a public ``http(s)`` link or a ``data:image/...;base64,...`` value.
        model: AcruxCore public model name; defaults to ``config.EXTRACT_MODEL``.
        alias: prompt alias, ``production`` or ``staging``; defaults to ``config.PROMPT_ALIAS``.
        tags: extra trace tags.

    Raises:
        ValueError: the model reply was not valid JSON.
    """
    model = model or config.EXTRACT_MODEL
    alias = alias or config.PROMPT_ALIAS
    async with acrux.AcruxCore(api_key=config.ACRUXCORE_API_KEY, base_url=config.ACRUXCORE_BASE_URL) as hub:
        rendered = await hub.prompts.render(config.PROMPT_NAME, alias, {"receipt_url": receipt_url})
        started = time.perf_counter()
        result = await hub.gateway.chat(
            model, rendered.messages, temperature=0, max_tokens=500,
            prompt_version_id=rendered.version_id, variables=rendered.variables,
            trace={"name": "extract-receipt-fields", "tags": ["receipt-sorter", alias, *(tags or [])], "metadata": {"model": model}},
        )
        seconds = time.perf_counter() - started
        fields = parse_fields(result.content or "")
        return Fields(**fields, trace_id=result.gateway.trace_id, raw=result.content or "", seconds=seconds)
