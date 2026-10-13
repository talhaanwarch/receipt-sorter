"""Score an AcruxCore evaluation run by exact match, with no judge involved.

    python scripts/score_run.py <run-id> [<run-id> ...]

With several runs it also prints, per cell, the lowest and highest "all 3" count, the median
seconds per receipt and the dollars per 1,000 receipts. A reply that failed at the provider
(a 429, for example) is counted as an error, not as a wrong answer.

For every (prompt version, model) cell it reads each example's raw model output and
compares company, date and total with the labels. The company must match after
removing case and punctuation; the date must be the same day; the total must be the
same number (31.9 equals 31.90). A lenient company match, one name inside the other, is
counted too, because labels hold legal names. Also prints how many examples the judge scored.
"""
import asyncio
import statistics
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


async def span_of(hub: acrux.AcruxCore, trace_id: str | None, sem: asyncio.Semaphore) -> dict:
    """The gateway call's latency (ms) and cost (USD), from the first llm span of an example's trace."""
    if not trace_id:
        return {}
    async with sem:
        spans = (await hub.traces.get(trace_id)).spans
    llm = next((s for s in spans if s.kind == "llm"), None)
    return {"ms": llm.latency_ms, "usd": llm.cost_usd} if llm else {}


async def score(run_id: str) -> list[dict]:
    rows = []
    sem = asyncio.Semaphore(10)
    async with acrux.AcruxCore(api_key=config.ACRUXCORE_API_KEY, base_url=config.ACRUXCORE_BASE_URL) as hub:
        report = await hub.runs.get_report(run_id)
        for cell in report.cells:
            detail = dataclasses.asdict(await hub.runs.get_cell(run_id, cell["cellKey"]))
            spans = await asyncio.gather(*(span_of(hub, ex.get("traceId"), sem) for ex in detail["examples"]))
            tally = {"n": 0, "company": 0, "company_lenient": 0, "date": 0, "total": 0, "all": 0, "unparsed": 0, "errors": 0}
            for ex in detail["examples"]:
                want = expected(ex["criteria"])
                tally["n"] += 1
                if ex.get("errorMessage"):
                    tally["errors"] += 1
                    continue
                try:
                    got = parse_fields(ex.get("output") or "")
                except (ValueError, json.JSONDecodeError):
                    tally["unparsed"] += 1
                    continue
                c, d, t = squash(got["company"]) == squash(want["company"]), same_date(got["date"], want["date"]), same_total(got["total"], want["total"])
                tally["company"] += c; tally["company_lenient"] += lenient(got["company"], want["company"]); tally["date"] += d; tally["total"] += t; tally["all"] += c and d and t
            secs = [float(s["ms"]) / 1000 for s in spans if s.get("ms") is not None]
            usd = [float(s["usd"]) for s in spans if s.get("usd") is not None]
            rows.append({"run": run_id, "cell": cell["cellKey"], "judge_scored": cell["scoredCount"], "judge_avg": cell["avgScore"], **tally,
                         "median_s": statistics.median(secs) if secs else None, "usd_per_1000": 1000 * sum(usd) / len(usd) if usd else None})
    return rows


def summarize(rows: list[dict]) -> None:
    print(f"\n{'cell':<22}{'runs':>5}{'all 3 (min-max)':>17}{'lenient co.':>13}{'errors':>8}{'median s':>10}{'$/1000':>9}{'judge avg':>11}")
    for cell in dict.fromkeys(r["cell"] for r in rows):
        rs = [r for r in rows if r["cell"] == cell]
        alls = [r["all"] for r in rs]
        med = statistics.median(r["median_s"] for r in rs if r["median_s"] is not None)
        usd = statistics.mean(r["usd_per_1000"] for r in rs if r["usd_per_1000"] is not None)
        judge = statistics.mean(r["judge_avg"] for r in rs if r["judge_avg"] is not None)
        lenient_min = min(r["company_lenient"] for r in rs)
        print(f"{cell:<22}{len(rs):>5}{min(alls):>10}-{max(alls):<3}/{rs[0]['n']}{lenient_min:>10}/{rs[0]['n']}{sum(r['errors'] for r in rs):>8}{med:>10.1f}{usd:>9.2f}{judge:>11.1f}")


async def main(run_ids: list[str]) -> None:
    all_rows = [r for rid in run_ids for r in await score(rid)]
    print(f"{'cell':<26}{'company':>9}{'date':>7}{'total':>7}{'all 3':>7}{'errors':>8}{'no JSON':>9}{'judged':>8}{'judge avg':>11}")
    for r in all_rows:
        n = r["n"]
        avg = "-" if r["judge_avg"] is None else f"{r['judge_avg']:.1f}"
        print(f"{r['cell']:<26}{r['company']:>6}/{n}{r['date']:>4}/{n}{r['total']:>4}/{n}{r['all']:>4}/{n}{r['errors']:>8}{r['unparsed']:>9}{r['judge_scored']:>5}/{n}{avg:>11}")
    if len(run_ids) > 1:
        summarize(all_rows)
    out = config.DATA / "results" / "run-scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=1))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
