"""Backfill real page numbers into the eval set from the live Chroma index.

run.py's assert_ready() refuses any chunk with page <= 0, so the sections left
at page 0 must be resolved before the eval can run. This reads the page from
your fmvss_571_208 Chroma metadata (no API key, no embedding — a pure local
read) and patches evals/v1_eval_set.json in place, writing a .bak first.

Run from the project root:
    python evals/fill_pages.py
    python evals/fill_pages.py --persist data/chroma --collection fmvss_571_208
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = Path(__file__).resolve().parent / "v1_eval_set.json"


def _norm(sub: str) -> str:
    """Normalise a subsection label for fuzzy matching: drop (a)/(b), case, trailing dot."""
    return re.sub(r"\(.*?\)", "", sub).strip().rstrip(".").lower()


def build_page_maps(persist: str, collection_name: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return (exact subsection->page, normalised subsection->page) from Chroma metadata."""
    client = chromadb.PersistentClient(path=persist)
    coll = client.get_collection(collection_name)
    got = coll.get(include=["metadatas"], limit=max(coll.count(), 1))
    metas = got.get("metadatas") or []

    exact: dict[str, int] = {}
    norm: dict[str, int] = {}
    for m in metas:
        sub = str(m.get("subsection", "")).strip()
        if not sub:
            continue
        try:
            page = int(round(float(m.get("page", 0))))
        except (TypeError, ValueError):
            continue
        if page <= 0:
            continue
        # keep the smallest page seen for a subsection (its start page)
        if sub not in exact or page < exact[sub]:
            exact[sub] = page
        nk = _norm(sub)
        if nk and (nk not in norm or page < norm[nk]):
            norm[nk] = page
    return exact, norm


def resolve(sub: str, exact: dict[str, int], norm: dict[str, int]) -> int | None:
    if sub in exact:
        return exact[sub]
    nk = _norm(sub)
    return norm.get(nk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist", default=str(PROJECT_ROOT / "data" / "chroma"))
    ap.add_argument("--collection", default="fmvss_571_208")
    args = ap.parse_args()

    if not EVAL_PATH.exists():
        raise SystemExit(f"eval set not found: {EVAL_PATH}")

    exact, norm = build_page_maps(args.persist, args.collection)
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    resolved: list[tuple[str, str, int]] = []
    unresolved: list[tuple[str, str]] = []
    for it in items:
        for c in it["chunks"]:
            if int(c.get("page", 0)) > 0:
                continue
            sub = c["subsection"]
            page = resolve(sub, exact, norm)
            if page is None:
                unresolved.append((it["id"], sub))
            else:
                c["page"] = page
                resolved.append((it["id"], sub, page))

    shutil.copyfile(EVAL_PATH, EVAL_PATH.with_suffix(".json.bak"))
    EVAL_PATH.write_text(json.dumps(items, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"resolved {len(resolved)} chunk page(s):")
    for iid, sub, pg in resolved:
        print(f"  {iid:<12} {sub:<14} -> p.{pg}")

    if unresolved:
        print(f"\nCOULD NOT MATCH {len(unresolved)} — fill these by hand in {EVAL_PATH.name}:")
        for iid, sub in unresolved:
            print(f"  {iid:<12} {sub}")
        print("\n  subsection labels present in Chroma (for reference):")
        print("  " + ", ".join(sorted(exact)))
        sys.exit(1)

    print("\nAll page-0 chunks resolved. run.py assert_ready() will now pass.")


if __name__ == "__main__":
    main()