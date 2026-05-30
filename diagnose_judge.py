"""Show the judge's raw verdict on direct_003 in three forms, to prove what it
is flagging as unsupported.

Run from the project root:
    python diagnose_judge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env", override=True)

from compliance_rag.eval.harness import load_eval_set       # noqa: E402
from compliance_rag.eval.scoring import _JUDGE_SYSTEM, _TAG_RE  # noqa: E402
from compliance_rag.generate import DISCLAIMER, generate_answer  # noqa: E402
from compliance_rag.llm.factory import build_llm             # noqa: E402

EVAL_PATH = ROOT / "evals" / "v1_eval_set.json"


def judge_raw(judge, context: str, answer: str) -> str:
    return judge.generate(
        _JUDGE_SYSTEM,
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}",
        temperature=0.0,
        max_tokens=512,
    ).strip()


def main() -> None:
    item = {it.id: it for it in load_eval_set(EVAL_PATH)}["direct_003"]
    gen = build_llm("claude-haiku-4-5-20251001")
    judge = build_llm("claude-sonnet-4-6")
    context = "\n\n".join(c.text for c in item.chunks)
    answer = generate_answer(item.question, item.chunks, llm=gen)

    variants = {
        "1) raw answer (tags + disclaimer)": answer,
        "2) disclaimer stripped": answer.replace(DISCLAIMER, "").strip(),
        "3) tags + disclaimer stripped": _TAG_RE.sub("", answer).replace(DISCLAIMER, "").strip(),
    }

    print("CONTEXT the judge sees (no section/page/date in it):")
    print(context)
    print("=" * 72)
    for label, variant in variants.items():
        print(f"\n### {label}")
        print("ANSWER sent to judge:")
        print(variant)
        print("JUDGE RAW VERDICT:")
        print(judge_raw(judge, context, variant))
        print("-" * 72)


if __name__ == "__main__":
    main()