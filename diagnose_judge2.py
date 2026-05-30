"""Verify the real faithfulness fix.

Judges the answer (minus the disclaimer) against a CONTEXT that includes each
chunk's citation tag — the same context the model saw during generation. Runs
the judge twice per item to confirm the verdict is now stable, and contrasts it
with the old text-only context.

Run from the project root:
    python diagnose_judge2.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env", override=True)

from compliance_rag.eval.harness import load_eval_set            # noqa: E402
from compliance_rag.generate import DISCLAIMER, generate_answer, render_citation  # noqa: E402
from compliance_rag.llm.factory import build_llm                 # noqa: E402

EVAL_PATH = ROOT / "evals" / "v1_eval_set.json"
IDS = ["direct_003", "synth_001"]

NEW_JUDGE_SYSTEM = (
    "You are a strict grader. Given CONTEXT and an ANSWER, decide whether every "
    "substantive factual claim in the ANSWER is supported by the CONTEXT. The CONTEXT "
    "includes a source citation tag (regulation, section, page, effective date) for "
    "each passage; treat a section number, page number, effective date, or "
    "[Source: ...] tag in the ANSWER as supported when it appears in the CONTEXT. "
    'Respond with ONLY JSON: {"supported": true|false, "unsupported": ["..."]}. '
    "No prose, no fences."
)


def extract(raw: str) -> dict | None:
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    cands = [raw]
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        cands.append(m.group())
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def label(judge, context: str, answer: str) -> str:
    raw = judge.generate(
        NEW_JUDGE_SYSTEM,
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}",
        temperature=0.0,
        max_tokens=512,
    )
    v = extract(raw)
    return "supported" if (v and v.get("supported")) else "NOT supported"


def main() -> None:
    items = {it.id: it for it in load_eval_set(EVAL_PATH)}
    gen = build_llm("claude-haiku-4-5-20251001")
    judge = build_llm("claude-sonnet-4-6")

    for iid in IDS:
        it = items[iid]
        answer = generate_answer(it.question, it.chunks, llm=gen).replace(DISCLAIMER, "").strip()
        text_only = "\n\n".join(c.text for c in it.chunks)
        with_cites = "\n\n".join(f"{c.text}\n{render_citation(c)}" for c in it.chunks)

        print("=" * 72)
        print(iid)
        print(f"  text-only context          -> {label(judge, text_only, answer)}   (the old bug)")
        print(f"  +citations context  run 1  -> {label(judge, with_cites, answer)}")
        print(f"  +citations context  run 2  -> {label(judge, with_cites, answer)}")


if __name__ == "__main__":
    main()