"""Create what the project needs in AcruxCore. Safe to run again.

    python scripts/setup_acruxcore.py

1. Registers the extraction models on the team's OpenRouter connection.
2. Creates the `receipt-extract` prompt: version 1 on production, version 2 committed
   and put on the staging alias.
3. Creates the `receipt-sorter-sroie-50` dataset from the Gemma transcripts, one example
   per receipt, with the expected company, date and total as the grading criteria.
"""
import asyncio
import json
import sys
from pathlib import Path

import acruxcore as acrux
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from receipt_sorter import config, prompts  # noqa: E402

# public name -> (OpenRouter model, USD per 1M input tokens, USD per 1M output tokens)
MODELS = {
    "rs-gemma-4-31b": ("google/gemma-4-31b-it", 0.09, 0.34),
    "rs-qwen3-vl-30b": ("qwen/qwen3-vl-30b-a3b-instruct", 0.15, 0.60),
    "rs-qwen3-vl-8b": ("qwen/qwen3-vl-8b-instruct", 0.25, 0.75),
    "rs-gpt-5.4-mini": ("openai/gpt-5.4-mini", 0.75, 4.50),
}
DATASET = "receipt-sorter-sroie-50"
TRANSCRIPT_SOURCE = "gemma-4-31b"


async def register_models(http: httpx.AsyncClient) -> None:
    conns = (await http.get("/gateway/connections")).json()
    cred = next(c for c in conns if c["provider"] == "openai_compatible" and "openrouter" in c["config"].get("base_url", ""))
    have = {m["publicName"] for m in (await http.get("/gateway/models")).json()}
    for name, (upstream, p_in, p_out) in MODELS.items():
        if name in have:
            print(f"model  {name}: exists")
            continue
        r = await http.post("/gateway/models", json={
            "publicName": name, "upstreamModel": upstream, "credentialId": cred["id"],
            "inputPricePerM": p_in, "outputPricePerM": p_out,
        })
        r.raise_for_status()
        print(f"model  {name}: registered -> {upstream}")


async def ensure_prompt(hub: acrux.AcruxCore) -> str:
    found = [p for p in (await hub.prompts.list(search=config.PROMPT_NAME, limit=20)).data if p.name == config.PROMPT_NAME]
    if found:
        print(f"prompt {config.PROMPT_NAME}: exists")
        return found[0].id
    prompt = await hub.prompts.create(config.PROMPT_NAME, description="Pulls company, date and total out of a receipt's text.")
    v1 = await hub.prompts.commit_version(prompt.id, prompts.messages(1))
    await hub.prompts.promote_alias(prompt.id, "production", v1.version_number)
    v2 = await hub.prompts.commit_version(prompt.id, prompts.messages(2))
    await hub.prompts.promote_alias(prompt.id, "staging", v2.version_number)
    print(f"prompt {config.PROMPT_NAME}: v{v1.version_number} on production, v{v2.version_number} on staging")
    return prompt.id


def criteria(label: dict) -> str:
    return (
        f'Expected company "{label["company"]}", date "{label["date"]}", total "{label["total"]}". '
        "Score high only if the reply is a JSON object with exactly these three values: the company may differ "
        "in capitalisation or punctuation only; the date must be the same day; the total must be the same "
        "number. Any other company, date or total means a low score."
    )


async def ensure_dataset(hub: acrux.AcruxCore) -> None:
    existing = next((d for d in await hub.datasets.list() if d.name == DATASET), None)
    if existing and existing.example_count:
        print(f"dataset {DATASET}: exists with {existing.example_count} examples")
        return
    ds = existing or await hub.datasets.create(DATASET, overall_feedback="Three fields must all be exactly right.")
    labels = json.loads(config.LABELS.read_text())
    for label in labels:
        text = json.loads((config.TRANSCRIPTS / TRANSCRIPT_SOURCE / f"{label['id']}.json").read_text())["text"]
        await hub.datasets.add_example(ds.id, {"transcript": text}, criteria=criteria(label))
    print(f"dataset {DATASET}: {len(labels)} examples")


async def main() -> None:
    async with acrux.AcruxCore(api_key=config.ACRUXCORE_API_KEY, base_url=config.ACRUXCORE_BASE_URL) as hub:
        async with httpx.AsyncClient(base_url=config.ACRUXCORE_BASE_URL, headers={"Authorization": f"Bearer {config.ACRUXCORE_API_KEY}"}, timeout=60) as http:
            await register_models(http)
        await ensure_prompt(hub)
        await ensure_dataset(hub)


if __name__ == "__main__":
    asyncio.run(main())
