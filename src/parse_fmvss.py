"""Parse and chunk an FMVSS regulation PDF into citation-ready chunks.

Pipeline stage (skills.md Skill 1):  parse -> chunk
Chunking rules (skills.md Skill 2):  preserve S-number hierarchy, never
split a numbered paragraph mid-sentence, carry parent heading + page +
effective date in metadata.

Run on the real source:
    python parse_fmvss.py data/raw/fmvss_571-208_occupant-crash-protection_2024-09-01.pdf \\
        --section 571.208 --effective-date 2024-09-01 \\
        --source-url https://www.ecfr.gov/current/title-49/.../section-571.208 \\
        --out data/processed/fmvss_571_208_chunks.json --inspect 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger("parse_fmvss")

# ---------------------------------------------------------------------------
# Tunables (skills.md Skill 1: chunk size 1000 / overlap 200 as the fallback
# splitter only; structure-aware splitting is primary).
# ---------------------------------------------------------------------------
MAX_CHARS = 1000
OVERLAP_CHARS = 200

# Lines that are running headers, edition footers, or GPO typesetting
# boilerplate in CFR print PDFs. These carry no regulatory content and
# corrupt chunks if left in. Matched per-line, before newline collapse.
NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*\d{1,4}\s*$"),                       # bare page numbers
    re.compile(r"Nat.?l Highway Traffic Safety Admin", re.I),
    re.compile(r"49\s*CFR\s*Ch\.?\s*V", re.I),            # "... 49 CFR Ch. V (Edition)"
    re.compile(r"VerDate\b.*", re.I),                     # typesetting timestamp line
    re.compile(r"\bJkt\b|\bPO\s*0{3,}|\bFrm\b|\bFmt\b|\bSfmt\b"),
    re.compile(r"DSK\w+OFR|with \$\$_JOB", re.I),         # printer queue noise
    re.compile(r"^\s*§\s*\d+\.\d+\s*$"),                  # bare running "§ 571.208"
    re.compile(r"page\s+\d+\s+of\s+\d+", re.I),           # eCFR "page 35 of 134"
    re.compile(r"\(enhanced display\)", re.I),            # eCFR running footer
    re.compile(r"up to date as of", re.I),                # eCFR currency header
    re.compile(r"^Standard No\.\s*208[;:]?\s+Occupant crash protection", re.I),
    re.compile(r"^49 CFR 571\.208 S\d"),                  # eCFR self-anchor header
]

# A heading line begins with an S-number token: S1, S4, S4.1, S4.1.2.1 ...
# group(2) keeps the RAW remainder (incl. any glued punctuation) so we can
# tell a real heading ("S4.1 Passenger cars.") from a wrapped cross-reference
# ("S5.1 with front test dummies ...", "S5.1.2(a)(1), ...").
SECTION_RE = re.compile(r"^(S\d+(?:\.\d+)*)(.*)$")

# A parent section whose own text is only its title (shorter than this AND
# having S-number children) is dropped as a standalone chunk; the title still
# rides along as its children's parent_heading.
HEADING_TITLE_MAXLEN = 120

# Sentence terminator used to avoid mid-sentence splits when a section is
# longer than MAX_CHARS. Keeps decimal section refs (S5.1) intact.
SENTENCE_END_RE = re.compile(r"(?<=[a-z0-9\)\"'])[.;]\s+(?=[A-Z(])")


@dataclass
class Chunk:
    """One retrieval unit. Field order matches skills.md Skill 2 schema."""

    text: str
    regulation: str
    section: str            # e.g. "571.208"
    subsection: str         # most-granular S-number, e.g. "S4.1.2.1"
    page: int
    effective_date: str
    source_url: str
    chunk_id: str
    parent_heading: str = ""
    part_index: int = 0     # >0 only when one section was split for length


@dataclass
class _Block:
    """An intermediate text block with its source page and position."""

    text: str
    page: int
    x0: float
    y0: float


# ---------------------------------------------------------------------------
# 1. Extraction (handles the two-column CFR print layout)
# ---------------------------------------------------------------------------
def _ordered_blocks(page: "fitz.Page", page_no: int) -> list[_Block]:
    """Return text blocks in human reading order, two-column aware.

    CFR print PDFs typeset two columns per page. PyMuPDF's raw block order
    is not guaranteed to be column-major, so we split blocks at the page
    mid-x and read the left column top-to-bottom, then the right column.
    Single-column PDFs collapse to the left bucket and behave normally.
    """
    mid_x = page.rect.width / 2.0
    raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
    left: list[_Block] = []
    right: list[_Block] = []
    for x0, y0, _x1, _y1, text, _bno, btype in raw:
        if btype != 0 or not text.strip():
            continue  # skip image blocks and empties
        bucket = left if x0 < mid_x else right
        bucket.append(_Block(text=text, page=page_no, x0=x0, y0=y0))
    left.sort(key=lambda b: b.y0)
    right.sort(key=lambda b: b.y0)
    return left + right


def extract_lines(pdf_path: Path) -> list[tuple[str, int]]:
    """Extract (line, page_number) pairs in reading order across the PDF."""
    doc = fitz.open(pdf_path)
    pairs: list[tuple[str, int]] = []
    for page_no, page in enumerate(doc, start=1):
        for block in _ordered_blocks(page, page_no):
            for line in block.text.splitlines():
                pairs.append((line, block.page))
    doc.close()
    logger.info("extracted %d raw lines from %d pages", len(pairs), page_no)
    return pairs


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def _is_noise(line: str) -> bool:
    return any(p.search(line) for p in NOISE_PATTERNS)


def clean_lines(pairs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Drop header/footer/typesetting noise; keep page tags."""
    kept = [(ln, pg) for ln, pg in pairs if ln.strip() and not _is_noise(ln)]
    logger.info("dropped %d noise lines (%d -> %d)",
                len(pairs) - len(kept), len(pairs), len(kept))
    return kept


def _dehyphenate(text: str) -> str:
    """Rejoin words split by an end-of-line hyphen, then flatten newlines."""
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)  # "se-\nverity" -> "severity"
    text = re.sub(r"\s*\n\s*", " ", text)               # remaining breaks -> space
    return re.sub(r"\s{2,}", " ", text).strip()


# ---------------------------------------------------------------------------
# 3. Structure parsing: group lines under their most-granular S-number
# ---------------------------------------------------------------------------
@dataclass
class _Section:
    s_number: str
    page: int
    lines: list[str] = field(default_factory=list)


def _looks_like_heading(rest: str, s_number: str, seen: set[str]) -> bool:
    """True only for genuine section headings, not wrapped cross-references.

    Real heading:    S-number then a capitalised title ("S4.1 Passenger cars.")
    Cross-reference:  S-number glued to punctuation     ("S5.1.2(a)(1), ...")
                      or followed by a lowercase word    ("S7.2 of this standard")
    Regulations never define the same section twice, so a repeat is a ref too.
    """
    if s_number in seen:
        return False
    if rest and rest[0] in "(,;:":   # glued cross-ref: S5.1.2(a), S15,
        return False
    title = rest.lstrip(" .")
    if title and title[0].islower():  # mid-sentence ref: "S5.1 with", "S7.2 of"
        return False
    # An empty title is fine: top-level headings (e.g. "S5") put the number on
    # its own line with the title wrapping to the next line. A cross-reference
    # never sits alone on a line, so this does not reintroduce false headings.
    return True


def parse_sections(pairs: list[tuple[str, int]]) -> list[_Section]:
    """Split cleaned lines into sections keyed by genuine S-number headings."""
    sections: list[_Section] = []
    seen: set[str] = set()
    current: _Section | None = None
    rejected = 0
    for line, page in pairs:
        m = SECTION_RE.match(line.strip())
        is_heading = bool(m) and _looks_like_heading(m.group(2), m.group(1), seen)
        if is_heading:
            seen.add(m.group(1))
            current = _Section(s_number=m.group(1), page=page)
            sections.append(current)
            title = m.group(2).lstrip(" .").strip()
            if title:
                current.lines.append(title)
        elif current is not None:
            current.lines.append(line)   # body line (incl. cross-ref look-alikes)
        if m and not is_heading:
            rejected += 1
    logger.info("parsed %d sections (rejected %d cross-ref look-alikes)",
                len(sections), rejected)
    return sections


def _parent_of(s_number: str, known: set[str]) -> str:
    """Nearest ancestor S-number that exists in the document, else ''."""
    parts = s_number.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:cut])
        if candidate in known:
            return candidate
    return ""


# ---------------------------------------------------------------------------
# 4. Chunking (structure-aware; length-split only as a fallback)
# ---------------------------------------------------------------------------
def _hard_window(text: str) -> list[str]:
    """Last-resort fixed-width split for a single over-long sentence."""
    parts, step, i = [], MAX_CHARS - OVERLAP_CHARS, 0
    while i < len(text):
        parts.append(text[i:i + MAX_CHARS].strip())
        if i + MAX_CHARS >= len(text):
            break
        i += step
    return [p for p in parts if p]


def _split_long(text: str) -> list[str]:
    """Split text > MAX_CHARS on sentence boundaries; hard-window as fallback."""
    if len(text) <= MAX_CHARS:
        return [text]
    packed, buf = [], ""
    for sent in SENTENCE_END_RE.split(text):
        if buf and len(buf) + len(sent) + 1 > MAX_CHARS:
            packed.append(buf.strip())
            buf = (buf[-OVERLAP_CHARS:] + " " + sent).strip()  # carry overlap
        else:
            buf = f"{buf} {sent}".strip()
    if buf.strip():
        packed.append(buf.strip())
    out: list[str] = []
    for part in packed:
        out.extend(_hard_window(part) if len(part) > MAX_CHARS else [part])
    return out


def _slug(section: str, s_number: str, page: int) -> str:
    base = f"{section}_{s_number}".lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return f"fmvss_{base}_p{page}"


def _has_children(s_number: str, known: set[str]) -> bool:
    prefix = s_number + "."
    return any(other.startswith(prefix) for other in known)


def build_chunks(
    sections: list[_Section],
    *,
    regulation: str,
    section_code: str,
    effective_date: str,
    source_url: str,
) -> list[Chunk]:
    """Convert parsed sections into citation-ready chunks."""
    known = {s.s_number for s in sections}
    used_ids: set[str] = set()
    chunks: list[Chunk] = []
    dropped = 0
    for sec in sections:
        body = _dehyphenate("\n".join(sec.lines))
        if not body:
            continue
        # Skip a parent section whose text is only its title — the title is
        # carried on its children via parent_heading, so a standalone
        # title-only chunk would just be retrieval noise.
        if len(body) < HEADING_TITLE_MAXLEN and _has_children(sec.s_number, known):
            dropped += 1
            continue
        parent = _parent_of(sec.s_number, known)
        for i, part in enumerate(_split_long(body)):
            cid = _slug(section_code, sec.s_number, sec.page)
            if i:
                cid = f"{cid}_part{i + 1}"
            while cid in used_ids:                  # guarantee uniqueness
                cid = f"{cid}_dup"
                logger.warning("duplicate chunk_id resolved -> %s", cid)
            used_ids.add(cid)
            chunks.append(Chunk(
                text=part,
                regulation=regulation,
                section=section_code,
                subsection=sec.s_number,
                page=sec.page,
                effective_date=effective_date,
                source_url=source_url,
                chunk_id=cid,
                parent_heading=parent,
                part_index=i,
            ))
    logger.info("built %d chunks from %d sections (dropped %d title-only parents)",
                len(chunks), len(sections), dropped)
    return chunks


# ---------------------------------------------------------------------------
# 5. Inspection (the human-facing report for Phase 2 Step 1)
# ---------------------------------------------------------------------------
def inspect(chunks: list[Chunk], n: int, seed: int = 7) -> None:
    """Print n random chunk samples plus quick health stats to stdout."""
    random.seed(seed)
    sample = random.sample(chunks, min(n, len(chunks)))
    lengths = [len(c.text) for c in chunks]
    tiny = sum(1 for n_ in lengths if n_ < 80)
    oversize = sum(1 for n_ in lengths if n_ > MAX_CHARS)
    no_parent = sum(1 for c in chunks if not c.parent_heading and "." in c.subsection)

    print("=" * 78)
    print(f"CHUNK HEALTH:  total={len(chunks)}  "
          f"mean_len={sum(lengths) // len(lengths)}  "
          f"min={min(lengths)}  max={max(lengths)}")
    print(f"  flags:  tiny(<80 chars)={tiny}  oversize(>{MAX_CHARS})={oversize}  "
          f"missing_parent={no_parent}")
    print("=" * 78)
    for i, c in enumerate(sample, 1):
        print(f"\n[{i}] chunk_id={c.chunk_id}")
        print(f"    subsection={c.subsection}  parent={c.parent_heading or '-'}  "
              f"page={c.page}  len={len(c.text)}  part={c.part_index}")
        preview = c.text if len(c.text) <= 320 else c.text[:317] + "..."
        print(f"    text: {preview}")
    print("\n" + "=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> list[Chunk]:
    pairs = clean_lines(extract_lines(Path(args.pdf)))
    sections = parse_sections(pairs)
    chunks = build_chunks(
        sections,
        regulation=args.regulation,
        section_code=args.section,
        effective_date=args.effective_date,
        source_url=args.source_url,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(c) for c in chunks], indent=2))
        logger.info("wrote %d chunks -> %s", len(chunks), out)
    if args.inspect:
        inspect(chunks, args.inspect)
    return chunks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse + chunk an FMVSS PDF.")
    p.add_argument("pdf", help="path to the source regulation PDF")
    p.add_argument("--regulation", default="FMVSS 571")
    p.add_argument("--section", default="571.208")
    p.add_argument("--effective-date", default="UNKNOWN",
                   help="ISO date of this regulation snapshot")
    p.add_argument("--source-url", default="")
    p.add_argument("--out", default="", help="optional JSON output path")
    p.add_argument("--inspect", type=int, default=0,
                   help="print N random chunk samples")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(_build_parser().parse_args())
