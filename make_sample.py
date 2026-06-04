"""Create a small public sample from the full evaluation gold-set.

Reads ``evals/v1_eval_set.json`` and writes ``evals/sample_eval_set.json``
containing ~6 representative items (one per category where possible, so a
bait/hallucination case is included). Run from the repository root:

    python make_sample.py
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("evals/v1_eval_set.json")
DST = Path("evals/sample_eval_set.json")
SAMPLE_SIZE = 6


def category(item: object) -> str:
    """Best-effort category label for an eval item, for variety selection."""
    if isinstance(item, dict):
        for key in ("category", "type", "kind", "tag"):
            value = item.get(key)
            if value:
                return str(value)
        for key in ("id", "qid", "name"):  # e.g. "edge_001" -> "edge"
            value = item.get(key)
            if isinstance(value, str) and "_" in value:
                return value.split("_")[0]
    return ""


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Not found: {SRC}  (run this from the repo root)")

    data = json.loads(SRC.read_text(encoding="utf-8"))

    # Locate the list of items whether top-level or nested under a key.
    wrap_key: str | None = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        wrap_key = next((k for k, v in data.items() if isinstance(v, list)), None)
        items = data.get(wrap_key, []) if wrap_key else []
    else:
        items = []

    if not items:
        raise SystemExit("Could not find a list of eval items in the file.")

    # One item per distinct category first, then top up to SAMPLE_SIZE.
    sample: list = []
    seen: set[str] = set()
    for item in items:
        label = category(item)
        if label not in seen:
            seen.add(label)
            sample.append(item)
        if len(sample) >= SAMPLE_SIZE:
            break
    for item in items:
        if len(sample) >= SAMPLE_SIZE:
            break
        if item not in sample:
            sample.append(item)

    out = (
        sample
        if wrap_key is None
        else {**{k: v for k, v in data.items() if k != wrap_key}, wrap_key: sample}
    )
    DST.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {DST}: {len(sample)} of {len(items)} items; categories={sorted(seen)}")


if __name__ == "__main__":
    main()