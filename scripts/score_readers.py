"""Stage 1 scoring: does each reader's transcript contain the true company, date and total?

    python scripts/score_readers.py

A field counts as found when its value appears in the transcript. This measures the
vision step on its own, before any extraction prompt is involved.
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from receipt_sorter import config  # noqa: E402


def squash(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def has_company(text: str, company: str) -> bool:
    return squash(company) in squash(text)


def has_date(text: str, date: str) -> bool:
    d, m, y = date.split("/")
    pattern = rf"\b0?{int(d)}\s*[/.\-]\s*0?{int(m)}\s*[/.\-]\s*(?:{y}|{y[2:]})\b"
    return re.search(pattern, text) is not None


def has_total(text: str, total: str) -> bool:
    want = float(re.sub(r"[^0-9.]", "", total))
    for tok in re.findall(r"\d[\d,]*\.\d{1,2}|\d+", text):
        try:
            if abs(float(tok.replace(",", "")) - want) < 0.005:
                return True
        except ValueError:
            pass
    return False


def main() -> None:
    labels = json.loads(config.LABELS.read_text())
    print(f"{'reader':<14}{'company':>9}{'date':>7}{'total':>7}{'all 3':>7}{'sec/receipt':>13}{'$ / 1000':>10}")
    rows = {}
    for rdir in sorted(config.TRANSCRIPTS.iterdir()):
        got = {"company": 0, "date": 0, "total": 0, "all": 0}
        secs, cost = [], 0.0
        for lab in labels:
            r = json.loads((rdir / f"{lab['id']}.json").read_text())
            c, d, t = has_company(r["text"], lab["company"]), has_date(r["text"], lab["date"]), has_total(r["text"], lab["total"])
            got["company"] += c; got["date"] += d; got["total"] += t; got["all"] += c and d and t
            secs.append(r["seconds"]); cost += r["cost_usd"] or 0
        n = len(labels)
        rows[rdir.name] = {**got, "n": n, "median_seconds": statistics.median(secs), "usd_per_1000": cost / n * 1000}
        print(f"{rdir.name:<14}{got['company']:>6}/{n}{got['date']:>4}/{n}{got['total']:>4}/{n}{got['all']:>4}/{n}{statistics.median(secs):>12.1f}s{cost / n * 1000:>10.2f}")
    out = config.DATA / "results" / "reader-scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
