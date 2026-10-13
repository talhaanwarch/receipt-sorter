"""Score an AcruxCore evaluation run by exact match, with no judge involved.

    python scripts/score_run.py <run-id> [<run-id> ...]

For every (prompt version, model) cell it reads each example's raw model output and
compares company, date and total with the labels. The company must match after
removing case and punctuation; the date must be the same day; the total must be the
same number (31.9 equals 31.90). A lenient company match, one name inside the other, is
counted too, because labels hold legal names. Also prints how many examples the judge scored.
"""
import asyncio
import dataclasses
import json
import re
import sys
from pathlib import Path

import acruxcore as acrux

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from receipt_sorter import config  # noqa: E402
from receipt_sorter.extract import parse_fields  # noqa: E402


def squash(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def lenient(a: str, b: str) -> bool:
    a, b = squash(a), squash(b)
    return bool(a) and bool(b) and (a in b or b in a)


def same_date(a: str, b: str) -> bool:
    pa, pb = (re.findall(r"\d+", x) for x in (a, b))
    if len(pa) != 3 or len(pb) != 3:
        return False
    norm = lambda p: (int(p[0]), int(p[1]), int(p[2]) % 100 if len(p[2]) == 4 else int(p[2]))
    return norm(pa) == norm(pb)


def same_total(a: str, b: str) -> bool:
    try:
        return abs(float(re.sub(r"[^0-9.]", "", a)) - float(re.sub(r"[^0-9.]", "", b))) < 0.005
    except ValueError:
        return False


def expected(criteria: str) -> dict[str, str]:
    m = re.search(r'Expected company "(.*?)", date "(.*?)", total "(.*?)"\.', criteria)
    return {"company": m.group(1), "date": m.group(2), "total": m.group(3)}


async def score(run_id: str) -> list[dict]:
    rows = []
    async with acrux.AcruxCore(api_key=config.ACRUXCORE_API_KEY, base_url=config.ACRUXCORE_BASE_URL) as hub:
        report = await hub.runs.get_report(run_id)
        for cell in report.cells:
            detail = dataclasses.asdict(await hub.runs.get_cell(run_id, cell["cellKey"]))
            tally = {"n": 0, "company": 0, "company_lenient": 0, "date": 0, "total": 0, "all": 0, "unparsed": 0}
            for ex in detail["examples"]:
                want = expected(ex["criteria"])
                tally["n"] += 1
                try:
                    got = parse_fields(ex.get("output") or "")
                except (ValueError, json.JSONDecodeError):
                    tally["unparsed"] += 1
                    continue
                c, d, t = squash(got["company"]) == squash(want["company"]), same_date(got["date"], want["date"]), same_total(got["total"], want["total"])
                tally["company"] += c; tally["company_lenient"] += lenient(got["company"], want["company"]); tally["date"] += d; tally["total"] += t; tally["all"] += c and d and t
            rows.append({"run": run_id, "cell": cell["cellKey"], "judge_scored": cell["scoredCount"], "judge_avg": cell["avgScore"], **tally})
    return rows


async def main(run_ids: list[str]) -> None:
    all_rows = [r for rid in run_ids for r in await score(rid)]
    print(f"{'cell':<26}{'company':>9}{'date':>7}{'total':>7}{'all 3':>7}{'no JSON':>9}{'judged':>8}{'judge avg':>11}")
    for r in all_rows:
        n = r["n"]
        avg = "-" if r["judge_avg"] is None else f"{r['judge_avg']:.1f}"
        print(f"{r['cell']:<26}{r['company']:>6}/{n}{r['date']:>4}/{n}{r['total']:>4}/{n}{r['all']:>4}/{n}{r['unparsed']:>9}{r['judge_scored']:>5}/{n}{avg:>11}")
    out = config.DATA / "results" / "run-scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=1))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
