
import json
import sys
from pathlib import Path

CHUNKS_FILE = Path("data/processed/fmvss_571_208_chunks.json")


def main() -> None:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    if len(sys.argv) == 3 and sys.argv[1] == "get":
        chunk_id = sys.argv[2]
        c = next((x for x in chunks if x["chunk_id"] == chunk_id), None)
        print(c["text"] if c else f"Not found: {chunk_id}")
        return

    keyword = sys.argv[1] if len(sys.argv) > 1 else "HIC"
    hits = [c for c in chunks if keyword.lower() in c["text"].lower()]
    print(f"Found {len(hits)} chunk(s) for '{keyword}':\n")
    for c in hits[:5]:
        print("---")
        print("chunk_id   :", c["chunk_id"])
        print("subsection :", c["subsection"])
        print("page       :", c["page"])
        print("text       :", c["text"][:300])
        print()


if __name__ == "__main__":
    main()
