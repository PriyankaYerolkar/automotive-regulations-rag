# CLAUDE.md

Operating rules for Claude Code in this repository. These are binding defaults.
To deviate, the human will say so explicitly in the prompt.

---

## Project

Automotive Regulations RAG — a focused, citation-grounded Q&A tool over
public automotive regulatory documents (FMVSS first). Built solo, as a
portfolio + future-SaaS project. The thesis: Mechanical/Automotive
engineering + GenAI is the differentiator. Code stays at that intersection,
never at "generic AI tool."

**V1 scope is FMVSS §571.208 ONLY** (Occupant Crash Protection — frontal
crash, airbags, seat belt systems), ~60 pages. NOT all of Part 571 (~800
pages). Do not widen scope without an explicit instruction. Widening V1 was
already considered and rejected.

---

## Hard constraints (never violate)

- **No copyrighted material.** Public-domain sources only: FMVSS (NHTSA/eCFR),
  NHTSA recalls + TSBs, AIS (ARAI), Bharat Stage (CPCB/MoRTH), NCAP, EPA
  emission data. Never scrape or ingest Haynes/Chilton, OEM service manuals,
  paid databases, or paywalled SAE papers.
- **No answer without a citation.** Generation that produces a compliance
  answer with no source citation is a defect, not a feature.
- **Compliance disclaimer required** on any user-facing generated answer:
  "This is not legal advice. Verify against the official regulation before any
  compliance decision."
- **Never commit secrets.** `.env` is gitignored; document every var in
  `.env.example`. No API keys in code, tests, or fixtures.

---

## RAG pipeline standard

Canonical order for every RAG component:

`ingest → parse → chunk → embed → store → retrieve → rerank → generate → cite`

| Stage | Default | Notes |
|---|---|---|
| Parse PDF | PyMuPDF (fitz) | fall back to pdfplumber for table-heavy pages |
| Chunk | RecursiveCharacterTextSplitter, size 1000, overlap 200 | semantic chunking only after V1 ships |
| Embed | one model per index — never mix | choice deferred (text-embedding-3-small vs BGE-small) |
| Store | Chroma local; Qdrant Cloud free tier for deployed demo | |
| Retrieve | top-k = 5 | tune during eval |
| Rerank | MMR, lambda 0.5 | mandatory, not optional |
| Generate | temperature = 0.1 for factual answers | |
| Cite | paragraph-level, mandatory | see citation format below |

**Refuse these anti-patterns** (flag them, don't silently implement):
- Embedding before inspecting chunk quality on a sample of ~10 chunks
- Generating answers without citations
- Skipping the rerank step "to save time"
- Mixing two embedding models in one index

---

## Regulatory chunking rules

Regulatory PDFs are hierarchical (part → section → subsection → numbered
paragraph), not prose. Chunking must preserve structure.

- Keep section/subsection numbers in chunk metadata (e.g. "§571.208 S4.2.1")
- Never split a numbered paragraph mid-sentence
- Include the parent section heading in every child chunk's metadata
- Store source document name, page number, and effective date

Chunk metadata schema:
```json
{
  "text": "...",
  "regulation": "FMVSS 571",
  "section": "571.208",
  "subsection": "S4.2.1",
  "page": 12,
  "effective_date": "2024-09-01",
  "source_url": "https://www.ecfr.gov/...",
  "chunk_id": "fmvss_571_208_s4_2_1_p12"
}
```

Source PDF naming convention:
`fmvss_<section>_<title-slug>_<YYYY-MM-DD>.pdf`
(embeds regulation, section, semantic title, snapshot date — enables
historical diffs and metadata extraction)

---

## Citation format

Every generated answer cites the specific supporting paragraph:

`[Source: FMVSS §571.208 S4.2.1, p.12 — effective 2024-09-01]`

- Cite the paragraph that supports the claim, not the whole document
- If no retrieved chunk supports a claim, the model must say
  "I could not find this in the retrieved documents" — never guess
- Multi-source answers: cite each source after the relevant sentence,
  not in one footer
- Always append the not-legal-advice disclaimer

---

## Python coding standards

- Python 3.11+
- Type hints on every function signature
- Black formatting, Ruff linting (configured in pyproject.toml)
- No `print()` in `src/` — use the `logging` module
- No bare `except:`
- Functions ≤ 50 lines; refactor if longer
- Public functions need Google-style docstrings
- Tests in `tests/` mirror `src/` structure; one assertion per test where reasonable
- Dependency management via uv or poetry — no `requirements.txt`

---

## Repo layout

```
repo-name/
├── README.md
├── LICENSE                 # MIT
├── pyproject.toml          # pinned versions
├── .gitignore
├── .env.example            # documents every required env var
├── src/<package>/
├── tests/                  # mirrors src/
├── notebooks/              # exploratory only, never production
├── evals/v1_eval_set.json
├── data/                   # actual data gitignored except small samples
├── docs/architecture.md
└── .github/workflows/ci.yml
```

---

## Evaluation gate

No release (V1 demo, paid tier, new document set) without passing the eval
suite. Eval set lives in `evals/v1_eval_set.json`, version-controlled,
minimum 30 Q-A pairs across: direct lookup, multi-section synthesis, edge
cases, hallucination bait.

V1 demo pass thresholds (starting targets, tune after first run):
- Context recall ≥ 0.85
- Faithfulness ≥ 0.90
- Citation accuracy ≥ 0.95
- Zero hallucinations on the bait set

---

## How to work in this repo

- Prefer Plan mode for anything touching more than one file or >50 lines.
- Break large tasks into steps; do not scaffold the entire pipeline in one shot.
- When choosing an approach, surface 2–3 alternatives with tradeoffs rather
  than silently picking one.
- State uncertainty explicitly (tool versions, API features) — do not bluff.
- Comments only where non-obvious. Imports at the top.
