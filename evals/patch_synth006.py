"""Patch synth_006 to drop the seat-belt-warning sections that aren't in the index.

The original synth_006 cited S4.1.5.7 and S4.1.5.8. Your Chroma index shows
S4.1.5.7 exists only as children (S4.1.5.7.1 / S4.1.5.7.2 — the parent heading
was dropped), and S4.1.5.8 is not in the indexed snapshot at all. Rather than
cite sections that can't be grounded, this replaces synth_006 with a verified
two-section synthesis: the chest-deflection limit (S6.4) plus the 300 ms data
window (S4.11). Pages are the real ones already confirmed from your index.

Run from the project root, AFTER fill_pages.py:
    python evals/patch_synth006.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

EVAL_PATH = Path(__file__).resolve().parent / "v1_eval_set.json"

CHEST = (
    "Chest deflection. (a) Compressive deflection of the sternum relative to the "
    "spine shall not exceed 76 mm (3.0 in). (b) Compressive deflection of the "
    "sternum relative to the spine shall not exceed 63 mm (2.5 in)."
)
DURATION = (
    "Test duration for purpose of measuring injury criteria. (a) For all barrier "
    "crashes, the injury criteria specified in this standard shall be met when "
    "calculated based on data recorded for 300 milliseconds after the vehicle "
    "strikes the barrier."
)

NEW_SYNTH_006 = {
    "id": "synth_006",
    "category": "multi_section_synthesis",
    "question": (
        "Under \u00a7571.208, what is the maximum chest deflection, and over what "
        "time window must the crash data be evaluated?"
    ),
    "is_bait": False,
    "chunks": [
        {
            "text": CHEST,
            "regulation": "FMVSS",
            "section": "571.208",
            "subsection": "S6.4",
            "page": 37,
            "effective_date": "2026-05-15",
        },
        {
            "text": DURATION,
            "regulation": "FMVSS",
            "section": "571.208",
            "subsection": "S4.11",
            "page": 35,
            "effective_date": "2026-05-15",
        },
    ],
}


def main() -> None:
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    for i, it in enumerate(items):
        if it["id"] == "synth_006":
            items[i] = NEW_SYNTH_006
            break
    else:
        raise SystemExit("synth_006 not found in eval set")

    shutil.copyfile(EVAL_PATH, EVAL_PATH.with_suffix(".json.bak2"))
    EVAL_PATH.write_text(json.dumps(items, indent=4, ensure_ascii=False), encoding="utf-8")

    zeros = [it["id"] for it in items for c in it["chunks"] if int(c["page"]) <= 0]
    print("synth_006 replaced.")
    print("page-0 remaining:", zeros or "none")
    if not zeros:
        print("Clean. run.py assert_ready() will pass — go ahead and run it.")
    else:
        print("Still some page-0 chunks above — those need a page before run.py will start.")


if __name__ == "__main__":
    main()