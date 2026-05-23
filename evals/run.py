"""Run the V1 eval across Sonnet and Haiku; print a comparison table, log each run."""
import json
import logging
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from compliance_rag.eval.harness import EvalItem, ModelReport, load_eval_set, run_model

logging.basicConfig(level=logging.INFO)

EVAL_SET = Path(__file__).parent / "v1_eval_set.json"
RUNS_DIR = Path(__file__).parent / "runs"
JUDGE_MODEL = "claude-sonnet-4-6"
ARMS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def assert_ready(items: list[EvalItem]) -> None:
    """Refuse to run if any chunk still holds template placeholders."""
    bad: list[str] = []
    for item in items:
        for c in item.chunks:
            if ("FILL" in c.text or c.page <= 0
                    or not _ISO.match(c.effective_date)
                    or c.subsection.lower().endswith(".x")):
                bad.append(item.id)
                break
    if bad:
        raise ValueError(
            f"Placeholder chunks in items {sorted(set(bad))}. "
            "Fill real §571.208 content before running."
        )


def _persist(report: ModelReport, stamp: str) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    payload = asdict(report) | {"passed": report.passed, "judge": JUDGE_MODEL}
    (RUNS_DIR / f"{stamp}_{report.model}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _print_table(reports: list[ModelReport]) -> None:
    rows = [
        ("citation acc ≥0.95", lambda r: f"{r.citation_accuracy:.3f}"),
        ("faithfulness ≥0.90", lambda r: f"{r.faithfulness:.3f}"),
        ("bait failures (=0)", lambda r: str(len(r.bait_failures))),
        ("VERDICT", lambda r: "PASS" if r.passed else "FAIL"),
    ]
    print(f"{'metric':<22}" + "".join(f"{r.model:<34}" for r in reports))
    for label, fn in rows:
        print(f"{label:<22}" + "".join(f"{fn(r):<34}" for r in reports))


def main() -> None:
    items = load_eval_set(EVAL_SET)
    assert_ready(items)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports = [run_model(m, items, JUDGE_MODEL) for m in ARMS]
    for r in reports:
        _persist(r, stamp)
    _print_table(reports)


if __name__ == "__main__":
    main()