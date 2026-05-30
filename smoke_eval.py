"""Smoke-test two eval items end-to-end before the full run.py pass.

Drives generate_answer() directly with the gold chunks embedded in the eval
set, so it exercises the generation + citation + refusal path only (no
embeddings, no Chroma). Only ANTHROPIC_API_KEY is required.

Run from the repo root:
    python smoke_two.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

# Explicit-path, override=True load — the rule from the env-shadowing incident.
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from src.compliance_rag.generate import Chunk, generate_answer, render_citation

EVAL_PATH = Path("evals/v1_eval_set.json")
REFUSAL = "I could not find this in the retrieved documents."
ITEMS_TO_CHECK = ("direct_003", "bait_004")


def load_item(eval_id: str) -> dict:
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return next(it for it in data if it["id"] == eval_id)


def run_one(eval_id: str) -> tuple[dict, str]:
    item = load_item(eval_id)
    chunks = [Chunk(**c) for c in item["chunks"]]

    print("=" * 72)
    print(f"{item['id']}   [{item['category']}]   is_bait={item['is_bait']}")
    print(f"Q: {item['question']}")
    print("-" * 72)
    print("Chunks fed to the model (the model must copy these tags verbatim):")
    for c in chunks:
        print("   ", render_citation(c))
    print("-" * 72)

    answer = generate_answer(item["question"], chunks)
    print(answer)
    print("=" * 72)
    return item, answer


def verdict(item: dict, answer: str) -> None:
    if item["is_bait"]:
        refused = REFUSAL in answer
        print(f">>> BAIT CHECK [{item['id']}]: "
              f"{'PASS — refused' if refused else 'FAIL — did NOT refuse (hallucination gate broken)'}")
    else:
        tag = render_citation(Chunk(**item["chunks"][0]))
        cited = tag in answer
        refused = REFUSAL in answer
        ok = cited and not refused
        print(f">>> NORMAL CHECK [{item['id']}]: "
              f"{'PASS' if ok else 'INSPECT MANUALLY'}  "
              f"(citation tag present={cited}, refused={refused})")
    print()


if __name__ == "__main__":
    for eid in ITEMS_TO_CHECK:
        item, answer = run_one(eid)
        verdict(item, answer)
