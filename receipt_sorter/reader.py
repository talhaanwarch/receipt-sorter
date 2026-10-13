"""Stage 1: read a receipt image and return its text.

This call goes straight to OpenRouter. The AcruxCore gateway accepts text-only
messages today (issue #567), so the image never passes through it. Everything after
this stage, the extraction prompt and the evaluations, runs in AcruxCore.
"""
import base64
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import config

READ_PROMPT = (
    "Transcribe all the text on this receipt exactly as printed, top to bottom, one line per "
    "line. Keep numbers, dates and punctuation unchanged. Output the text only."
)


@dataclass
class Reading:
    text: str
    model: str
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None


async def read_receipt(image: Path, model: str | None = None, client: httpx.AsyncClient | None = None) -> Reading:
    """Reads one receipt image with a vision model on OpenRouter.

    Args:
        image: JPEG file of the receipt.
        model: OpenRouter model id; defaults to ``config.READER_MODEL``.
        client: shared HTTP client, so a batch reuses connections.

    Returns:
        The transcript with timing, token counts and the cost OpenRouter reports.

    Raises:
        httpx.HTTPStatusError: OpenRouter answered with an error status.
    """
    model = model or config.READER_MODEL
    data_url = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
    body = {
        "model": model,
        "temperature": 0,
        "usage": {"include": True},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": READ_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
    }
    owns = client is None
    client = client or httpx.AsyncClient(timeout=180)
    try:
        started = time.perf_counter()
        resp = await client.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
            json=body,
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - started
    finally:
        if owns:
            await client.aclose()
    payload = resp.json()
    usage = payload.get("usage", {})
    return Reading(
        text=(payload["choices"][0]["message"]["content"] or "").strip(),
        model=model,
        seconds=elapsed,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cost_usd=usage.get("cost"),
    )
