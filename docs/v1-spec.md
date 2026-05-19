# V1 Specification — Automotive Regulations RAG

**Status:** Locked. Created 2026-05-18. Changes require a dated entry in memory.md.
**Parent docs:** context.md, skills.md (Skills 1–4, 9), memory.md.

## 1. What V1 is, in one sentence
A public web demo where a homologation engineer types a plain-English question
about US frontal-crash occupant protection and gets a cited, paragraph-level
answer grounded only in FMVSS §571.208.

## 2. In scope

| Item | Specifically |
|---|---|
| Regulation | FMVSS §571.208 — Occupant Crash Protection, ~60 pages |
| Source | eCFR Title 49 §571.208, current consolidated version |
| Language | English only |
| Modality | Single-turn text Q&A → answer + paragraph-level citations |
| Audience | Homologation engineers, solo compliance consultants (Personas A, B) |
| Deployment | Public Hugging Face Space, free tier |

## 3. Out of scope (the trap list — scope creep is impossible if these are explicit)

| Excluded | Reason |
|---|---|
| Any other FMVSS standard (§571.213, §571.214, etc.) | V2 |
| AIS / Bharat Stage / UNECE | V3+ |
| Recall data, TSBs, NHTSA APIs | Separate follow-on project |
| Hindi or non-English | V3 |
| User-uploaded PDFs | V1 corpus is fixed |
| Multi-turn conversation memory | V2 |
| User accounts, auth, payments | Not until Phase 6 validates SaaS |
| Image / diagram / table extraction | Future; V1 is text-first |
| Mobile-optimized UI | Desktop demo first |

## 4. Success criteria (V1 does not ship until these clear)

Eval set per Skill 4: minimum 30 Q&A pairs, version-controlled in
`/evals/v1_eval_set.json`. Categories adapted for single-section scope:

| Category | Example | Count |
|---|---|---|
| Direct lookup | "What is the maximum HIC value in S6?" | 12 |
| Within-section synthesis | "How do S4 belt-only vs belt+airbag test requirements differ?" | 8 |
| Edge cases | "What does §571.208 say about out-of-position occupants?" | 5 |
| Hallucination bait | Plausible queries with no answer in §571.208 | 5 |

Pass thresholds (starting targets per Skill 4; retune after first eval run):

| Metric | Threshold | Source |
|---|---|---|
| Context recall | ≥ 0.85 | Ragas |
| Faithfulness | ≥ 0.90 | Ragas |
| Answer relevancy | ≥ 0.80 | Ragas |
| Citation accuracy (cited paragraph supports the claim) | ≥ 0.95 | Custom |
| Hallucination rate on bait set | 0 | Manual review |

Every answer must end with the Skill 3 disclaimer:
> "This is not legal advice. Verify against the official regulation before any compliance decision."

## 5. Definition of "V1 ships"
All five must be true:
1. Live HF Space at a public URL.
2. README links to the live Space, has a 30-second quickstart and a screenshot/GIF.
3. Eval suite runs from a single command and meets every threshold in §4.
4. At least one validation conversation confirms at least one target user would try it.
5. A LinkedIn launch post is drafted (publishing is Phase 5, not a ship gate).

## 6. Stack — locked vs deferred

| Layer | V1 choice | Status |
|---|---|---|
| Python | 3.11+ | Locked |
| PDF parsing | PyMuPDF primary, pdfplumber fallback on tables | Locked (Skill 1) |
| Chunking | Recursive 1000/200 with regulatory-aware metadata | Locked (Skill 2) |
| Vector store | Chroma local; Qdrant Cloud free tier for deployed Space | Locked |
| Retrieval | top-k=5, MMR rerank λ=0.5 | Locked |
| Embeddings | `text-embedding-3-small` (paid) vs BGE-small (open) | **Decide Phase 2** after chunk-quality on 100 §571.208 samples |
| LLM | Claude Sonnet vs GPT-4o-mini | **Decide Phase 3** on per-query cost over 50 eval queries |
| UI | Streamlit vs Gradio | **Decide Phase 4** on HF Spaces compatibility + dev speed |

## 7. Timeline (part-time, ~10–15 hrs/week)

| Phase | Deliverable | Estimate |
|---|---|---|
| 2. Data pipeline | §571.208 parsed, chunked + metadata, embedded, sample-inspected | 1.0 wk |
| 3. RAG core | Retrieve → rerank → generate with citations + disclaimer | 1.5 wk |
| 4. Demo UI | Streamlit/Gradio app deployed to HF Spaces | 1.0 wk |
| Eval + polish | Eval set written, thresholds met, README updated | 0.5 wk |
| **Total** | **V1 shipped** | **~4 weeks** |

Meets Skill 9's ≤4-week part-time ship rule. Slippage past 6 weeks triggers a
scope review, not a deadline extension.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| One hallucinated clause kills trust permanently | Paragraph-level citations mandatory; zero-tolerance on bait set |
| Scope creep ("just add §571.214") | This spec is the contract; changes go in memory.md with date + reason |
| HF Space cold-start latency at demo time | Pre-warm before scheduled demos; document the limitation in README |
| Embedding model regret at scale | Choose in Phase 2 with §571.208 quality test before any V2 expansion |

## 9. Open questions to close during V1
- Public name for the demo.
- 5 domain-name candidates collected before Phase 5 (LinkedIn launch).
- Publish chunked §571.208 as a standalone HF dataset card? (Lead-gen vs. effort.)