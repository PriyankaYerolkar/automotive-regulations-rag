"""Probe: is the appended disclaimer tanking the faithfulness score?

Runs the real generate path on a few items, then scores faithfulness TWICE:
once on the full answer (with the 'not legal advice' disclaimer) and once with
that disclaimer stripped out. If the stripped score jumps up, the disclaimer is
the culprit (suspect #1). If both stay low, it's the judge/parser (suspect #2).

Run from the project root (same folder that holds src\ and evals\):
    python probe_faithfulness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env", override=True)

from compliance_rag.eval.harness import load_eval_set           # noqa: E402
from compliance_rag.eval.scoring import faithfulness            # noqa: E402
from compliance_rag.generate import DISCLAIMER, generate_answer  # noqa: E402
from compliance_rag.llm.factory import build_llm                 # noqa: E402

EVAL_PATH = ROOT / "evals" / "v1_eval_set.json"
IDS = ["direct_003", "direct_001", "synth_001"]


def strip_disclaimer(text: str) -> str:
    return text.replace(DISCLAIMER, "").strip()


def main() -> None:
    items = {it.id: it for it in load_eval_set(EVAL_PATH)}
    gen = build_llm("claude-haiku-4-5-20251001")
    judge = build_llm("claude-sonnet-4-6")

    print(f"{'item':<12}{'with disclaimer':<18}{'without':<10}")
    print("-" * 40)
    for iid in IDS:
        it = items[iid]
        answer = generate_answer(it.question, it.chunks, llm=gen)
        with_d = faithfulness(answer, it.chunks, judge)
        without_d = faithfulness(strip_disclaimer(answer), it.chunks, judge)
        print(f"{iid:<12}{with_d:<18.3f}{without_d:<10.3f}")
        if iid == "direct_003":
            print("\n--- direct_003 full answer the judge saw ---")
            print(answer)
            print("--- end ---\n")

    print("\nIf 'without' is much higher than 'with' -> the disclaimer is the cause (#1).")
    print("If both stay low -> it's the judge/parser (#2); paste scoring.py and I'll look.")


if __name__ == "__main__":
    main()