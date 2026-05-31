"""Local smoke test for the RegCite demo's example questions.

Run from the PROJECT ROOT (the folder that contains src/ and data/):

    python smoke_test.py

Pass criteria:
  - the four in-scope questions return a cited answer (contains "[Source:")
    and are NOT the refusal string;
  - the out-of-scope question IS the refusal string.

Exit code is non-zero if anything fails, so you can wire it into CI later.
This makes a handful of cheap API calls (embedding + Haiku) per question.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root so the keys are present (pipeline.py also
# loads it on import; override=True makes the file authoritative).
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from src.compliance_rag.generate import render_citation  # noqa: E402
from src.compliance_rag.pipeline import answer_with_sources  # noqa: E402

IN_SCOPE = [
    "What is the HIC limit under FMVSS 571.208?",
    "What is the maximum femur load for the 5th percentile adult female dummy?",
    "What is the maximum chest acceleration permitted under FMVSS 571.208?",
    "What is the maximum HIC15 value permitted for the Hybrid III dummy?",
]
OUT_OF_SCOPE = "What does FMVSS 208 require for autonomous-vehicle sensors?"
REFUSAL = "I could not find this in the retrieved documents."


def _print_result(question: str, answer: str, chunks) -> None:
    print("\n" + "=" * 72)
    print("Q:", question)
    print("-" * 72)
    print(answer)
    print("-" * 72)
    print("retrieved sources:")
    for c in chunks:
        print("   ", render_citation(c))


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set - check your .env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set - check your .env")

    failures = 0

    for q in IN_SCOPE:
        res = answer_with_sources(q)
        cited = "[Source:" in res.answer
        refused = REFUSAL in res.answer
        ok = cited and not refused
        if not ok:
            failures += 1
        _print_result(q, res.answer, res.chunks)
        print(f"VERDICT: cited={cited} refused={refused} -> {'PASS' if ok else 'FAIL'}")

    res = answer_with_sources(OUT_OF_SCOPE)
    refused = REFUSAL in res.answer
    if not refused:
        failures += 1
    _print_result(OUT_OF_SCOPE + "  (expected: refuse)", res.answer, res.chunks)
    print(f"VERDICT: refused={refused} -> {'PASS' if refused else 'FAIL'}")

    print("\n" + "=" * 72)
    print(f"TOTAL FAILURES: {failures} / 5")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)