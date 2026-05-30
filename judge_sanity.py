"""Sanity check: confirm the patched faithfulness judge still catches a wrong answer.

Feeds the real faithfulness() a correct answer and a deliberately wrong one for
direct_003, and prints both scores. Expect: correct -> 1.0, wrong -> 0.0.
If the wrong answer scores 1.0, the judge has gone toothless and we must tighten it.

Run from the project root:
    python judge_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env", override=True)

from compliance_rag.eval.harness import load_eval_set       # noqa: E402
from compliance_rag.eval.scoring import faithfulness         # noqa: E402
from compliance_rag.generate import render_citation          # noqa: E402
from compliance_rag.llm.factory import build_llm             # noqa: E402

EVAL_PATH = ROOT / "evals" / "v1_eval_set.json"


def main() -> None:
    item = {it.id: it for it in load_eval_set(EVAL_PATH)}["direct_003"]
    judge = build_llm("claude-sonnet-4-6")
    tag = render_citation(item.chunks[0])

    correct = (
        "The maximum resultant thoracic acceleration is 60 g's, except for intervals "
        f"whose cumulative duration is not more than 3 milliseconds. {tag}"
    )
    wrong = (
        f"The maximum resultant thoracic acceleration is 95 g's, with no exception "
        f"permitted. {tag}"
    )

    print("correct answer -> faithfulness:", faithfulness(correct, item.chunks, judge), "(expect 1.0)")
    print("wrong answer   -> faithfulness:", faithfulness(wrong, item.chunks, judge), "(expect 0.0)")


if __name__ == "__main__":
    main()