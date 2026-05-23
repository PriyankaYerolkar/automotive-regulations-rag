"""Diagnose citation scoring: run ONE answerable item and show exactly what the model emitted."""
import logging
from pathlib import Path

from compliance_rag.eval.harness import load_eval_set
from compliance_rag.eval.scoring import emitted_tags
from compliance_rag.generate import generate_answer, render_citation
from compliance_rag.llm.factory import build_llm

logging.basicConfig(level=logging.INFO)


def main() -> None:
    items = load_eval_set(Path("evals/v1_eval_set.json"))
    item = next(i for i in items if not i.is_bait)
    answer = generate_answer(item.question, item.chunks, llm=build_llm("claude-sonnet-4-6"))

    expected = [render_citation(c) for c in item.chunks]
    got = emitted_tags(answer)

    print("QUESTION:", item.question)
    print("\n--- RAW ANSWER ---\n", answer)
    print("\n--- EXPECTED TAGS (rendered by code) ---")
    for t in expected:
        print(repr(t))            # repr() is the whole point — see below
    print("\n--- EMITTED TAGS (regex-extracted from answer) ---")
    for t in got:
        print(repr(t))
    print("\n--- MATCH CHECK ---")
    for t in got:
        print(("MATCH" if t in expected else "MISS "), repr(t))


if __name__ == "__main__":
    main()