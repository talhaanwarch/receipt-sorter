"""Stage 1 over the whole test set: read every receipt with each reader model.

    python scripts/read_receipts.py

Transcripts are cached in data/transcripts/<reader>/<id>.json, so a rerun only
pays for receipts that are missing.
"""
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from receipt_sorter import config, reader  # noqa: E402

READERS = {
    "gemma-4-31b": "google/gemma-4-31b-it",              # open weights
    "qwen3-vl-30b": "qwen/qwen3-vl-30b-a3b-instruct",    # open weights
    "gpt-5.4-mini": "openai/gpt-5.4-mini",               # closed baseline
}
CONCURRENCY = 6


async def main() -> None:
    labels = json.loads(config.LABELS.read_text())
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=240) as client:
        async def one(slug: str, model: str, rid: str) -> None:
            out = config.TRANSCRIPTS / slug / f"{rid}.json"
            if out.exists():
                return
            async with sem:
                for attempt in range(3):
                    try:
                        r = await reader.read_receipt(config.RECEIPTS / f"{rid}.jpg", model, client)
                        break
                    except httpx.HTTPError as e:
                        if attempt == 2:
                            print(f"FAILED {slug} {rid}: {e}")
                            return
                        await asyncio.sleep(3 * (attempt + 1))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(asdict(r), ensure_ascii=False))
            print(f"{slug} {rid} {r.seconds:.0f}s ${r.cost_usd or 0:.5f}", flush=True)

        await asyncio.gather(*(one(s, m, l["id"]) for s, m in READERS.items() for l in labels))


if __name__ == "__main__":
    asyncio.run(main())
