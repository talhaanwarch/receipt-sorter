"""Run the extraction prompt against the dataset in AcruxCore and save the report.

    python scripts/run_eval.py                 # both prompt versions on all four models, once
    python scripts/run_eval.py --repeat 3      # three runs: the same prompt scores a little differently each time
    python scripts/run_eval.py --versions 1    # prompt v1 only
    python scripts/run_eval.py --models rs-qwen3-vl-8b   # one model only, to run it again

The worker processes the run in the background; this script polls until it finishes
and writes the report to data/results/<run-id>.json.
"""
import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

import acruxcore as acrux

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from receipt_sorter import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_acruxcore import DATASET, MODELS  # noqa: E402


def to_plain(obj):
    """Dataclass tree -> plain dicts, so the report can be written as JSON."""
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


async def main(versions: list[int], repeat: int, models: list[str]) -> None:
    async with acrux.AcruxCore(api_key=config.ACRUXCORE_API_KEY, base_url=config.ACRUXCORE_BASE_URL) as hub:
        prompt = next(p for p in (await hub.prompts.list(search=config.PROMPT_NAME, limit=20)).data if p.name == config.PROMPT_NAME)
        by_number = {v.version_number: v.id for v in (await hub.prompts.list_versions(prompt.id)).data}
        dataset = next(d for d in await hub.datasets.list() if d.name == DATASET)
        for n in range(1, repeat + 1):
            exp = await hub.experiments.create(
                dataset.id, [by_number[v] for v in versions], models, prompt_id=prompt.id,
                name=f"receipts: repeat {n}/{repeat}, prompt v{','.join(map(str, versions))} x {len(models)} models",
            )
            run = await hub.experiments.start_run(exp.id)
            print(f"run {n}/{repeat}", run.run_id, run.status, flush=True)
            while True:
                detail = await hub.runs.get(run.run_id)
                if detail.status in ("succeeded", "failed", "cancelled"):
                    break
                await asyncio.sleep(10)
            report = await hub.runs.get_report(run.run_id)
            # A missing or broken judge model does not fail the run: every example just stays
            # unscored. Stop here instead of reporting a run that judged nothing.
            unscored = [c["cellKey"] for c in report.cells if not c.get("scoredCount")]
            if detail.status != "succeeded" or unscored:
                sys.exit(f"run {run.run_id} ended {detail.status}; cells with no judge scores: {unscored}. Open one example and read its reason.")
            out = config.DATA / "results" / f"{run.run_id}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"run": to_plain(detail), "report": to_plain(report)}, ensure_ascii=False, indent=1, default=str))
            print("saved", out, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=[m for m in MODELS if m.startswith("rs-")])
    args = ap.parse_args()
    asyncio.run(main(args.versions, args.repeat, args.models))
