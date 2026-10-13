"""Build the 50-receipt test set from the SROIE test split.

Downloads the parquet file from Hugging Face (about 190 MB, cached in .cache/),
picks 50 receipts with a fixed seed, writes the images to data/receipts/ and the
expected answers to data/labels.json.

Source: ICDAR 2019 SROIE, CC BY 4.0, via https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE
Images are not committed to git; run this script to recreate them.
"""
import io
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
URL = "https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE/resolve/main/data/test-00000-of-00001.parquet"
CACHE = ROOT / ".cache" / "sroie-test.parquet"
SAMPLE_SIZE = 50
SEED = 7
MAX_PER_COMPANY = 2
MAX_SIDE = 1600  # long side in px; keeps small print readable for a vision model


def fetch() -> Path:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    return CACHE


def parse_total(text: str) -> float | None:
    cleaned = re.sub(r"[^0-9.]", "", text.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def main() -> None:
    table = pq.read_table(fetch(), columns=["key", "entities", "image"]).to_pylist()
    usable = []
    for row in table:
        e = row["entities"]
        total = parse_total(e["total"] or "")
        if not (e["company"] and total and re.fullmatch(r"\d{2}/\d{2}/\d{4}", e["date"] or "")):
            continue
        usable.append({**row, "total_value": total})
    print(f"{len(usable)} of {len(table)} receipts have a company, a DD/MM/YYYY date and a numeric total")

    # Spread across the range of totals: sort, split into 5 bands, draw 10 from each.
    rng = random.Random(SEED)
    usable.sort(key=lambda r: r["total_value"])
    band = len(usable) // 5
    per_company: Counter[str] = Counter()
    picked = []
    for i in range(5):
        pool = usable[i * band:(i + 1) * band if i < 4 else len(usable)]
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            name = r["entities"]["company"].strip().upper()
            if per_company[name] >= MAX_PER_COMPANY:
                continue
            per_company[name] += 1
            picked.append(r)
            taken += 1
            if taken == SAMPLE_SIZE // 5:
                break

    out_dir = ROOT / "data" / "receipts"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for r in sorted(picked, key=lambda r: r["key"]):
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(out_dir / f"{r['key']}.jpg", quality=88)
        e = r["entities"]
        labels.append({"id": r["key"], "company": e["company"], "date": e["date"], "total": e["total"]})
    (ROOT / "data" / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(labels)} receipts and data/labels.json")


if __name__ == "__main__":
    main()
