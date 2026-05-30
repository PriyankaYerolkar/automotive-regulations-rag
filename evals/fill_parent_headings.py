"""Backfill parent_heading into the eval set from the live Chroma index.

The model now sees each chunk's parent section heading (generate._build_context),
so the eval chunks must carry the same parent_heading the production pipeline gets
from Chroma metadata. This reads parent_heading by subsection (a pure local read —
no API key) and writes it into evals/v1_eval_set.json, backing up to .bak3 first.

Run from the project root:
    python evals/fill_parent_headings.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = Path(__file__).resolve().parent / "v1_eval_set.json"


def _norm(sub: str) -> str:
    return re.sub(r"\(.*?\)", "", sub).strip().rstrip(".").lower()


def build_maps(persist: str, collection_name: str) -> tuple[dict[str, str], dict[str, str]]:
    client = chromadb.PersistentClient(path=persist)
    coll = client.get_collection(collection_name)
    got = coll.get(include=["metadatas"], limit=max(coll.count(), 1))
    metas = got.get("metadatas") or []
    exact: dict[str, str] = {}
    norm: dict[str, str] = {}
    for m in metas:
        sub = str(m.get("subsection", "")).strip()
        head = str(m.get("parent_heading", "")).strip()
        if not sub or not head:
            continue
        exact.setdefault(sub, head)
        norm.setdefault(_norm(sub), head)
    return exact, norm


def resolve(sub: str, exact: dict[str, str], norm: dict[str, str]) -> str | None:
    if sub in exact:
        return exact[sub]
    return norm.get(_norm(sub))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist", default=str(PROJECT_ROOT / "data" / "chroma"))
    ap.add_argument("--collection", default="fmvss_571_208")
    args = ap.parse_args()

    if not EVAL_PATH.exists():
        raise SystemExit(f"eval set not found: {EVAL_PATH}")

    exact, norm = build_maps(args.persist, args.collection)
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    filled: list[tuple[str, str, str]] = []
    missing: list[tuple[str, str]] = []
    for it in items:
        for c in it["chunks"]:
            head = resolve(c["subsection"], exact, norm)
            if head:
                c["parent_heading"] = head
                filled.append((it["id"], c["subsection"], head))
            else:
                c.setdefault("parent_heading", "")
                missing.append((it["id"], c["subsection"]))

    shutil.copyfile(EVAL_PATH, EVAL_PATH.with_suffix(".json.bak3"))
    EVAL_PATH.write_text(json.dumps(items, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"filled parent_heading on {len(filled)} chunk(s):")
    for iid, sub, head in filled:
        print(f"  {iid:<12} {sub:<14} -> {head[:60]}")
    if missing:
        print(f'\nno parent_heading in Chroma for {len(missing)} chunk(s) (left as ""):')
        for iid, sub in missing:
            print(f"  {iid:<12} {sub}")
    print("\nDone. Check that edge_001 and edge_002 now show a heading above.")


if __name__ == "__main__":
    main()