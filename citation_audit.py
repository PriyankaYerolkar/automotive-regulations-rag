"""Pinpoint Haiku's citation misses.

Runs Haiku over the answerable items, scores citation per item, and prints the
failing ones with emitted vs expected tags so you can see exactly what it does
wrong (skips a tag, drops the effective-date, merges two tags, etc.).

Run from the project root:
    python citation_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env", override=True)

from compliance_rag.eval.harness import load_eval_set            # noqa: E402
from compliance_rag.eval.scoring import citation_score, emitted_tags  # noqa: E402
from compliance_rag.generate import generate_answer, render_citation  # noqa: E402
from compliance_rag.llm.factory import build_llm                 # noqa: E402

EVAL_PATH = ROOT / "evals" / "v1_eval_set.json"
MODEL = "claude-haiku-4-5-20251001"


def main() -> None:
    items = [it for it in load_eval_set(EVAL_PATH) if not it.is_bait]
    llm = build_llm(MODEL)
    fails: list[str] = []
    for it in items:
        answer = generate_answer(it.question, it.chunks, llm=llm)
        if citation_score(answer, it.chunks) < 1.0:
            fails.append(it.id)
            print("=" * 72)
            print(f"FAIL  {it.id}  [{it.category}]")
            print("Q:", it.question)
            print("emitted tags :", emitted_tags(answer) or "(none)")
            print("expected tags:", [render_citation(c) for c in it.chunks])
            print("answer:\n" + answer)
    print("=" * 72)
    print(f"{len(items) - len(fails)}/{len(items)} citation-clean; failing: {fails or 'none'}")


if __name__ == "__main__":
    main()