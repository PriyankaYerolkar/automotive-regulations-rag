# Automotive Regulations RAG

A citation-grounded question-answering system for U.S. federal automotive safety regulations (FMVSS). Built for homologation engineers and compliance consultants who currently waste hours searching 1000-page PDFs to verify a single clause.

> **Status:** Pre-alpha. Scaffolding complete; V1 (FMVSS Part 571) in active development.

## Demo

Coming after V1 ships (Phase 4). This section will link to a live Hugging Face Space.

## What it does

Ask a natural-language question about FMVSS regulations. Get a precise answer with the specific section, subsection, and page cited — so you can verify against the official regulation in seconds, not hours.

Example query: *"What is the maximum allowable head injury criterion under FMVSS 208 for a 50th-percentile male dummy?"*

Example output:
> The maximum allowable HIC₁₅ is 700 for a 50th-percentile male Hybrid III dummy in frontal crash tests.
>
> [Source: FMVSS §571.208 S6.2(b), p.43 — effective 2024-09-01]
>
> *This is not legal advice. Verify against the official regulation before any compliance decision.*

## Quickstart

```bash
git clone https://github.com/priyankayerolkar/automotive-regulations-rag.git
cd automotive-regulations-rag
uv sync --extra dev
uv run pytest
```

## How it works

Standard RAG pipeline tuned for regulatory documents:

`PDF → parse (PyMuPDF) → hierarchical chunk → embed → vector store → retrieve top-k → rerank (MMR) → LLM generates with paragraph-level citation`

Architecture details in [docs/architecture.md](docs/architecture.md).

## Limitations

- V1 scope is FMVSS Part 571 only. AIS, Bharat Stage, and UNECE regulations come in V3+.
- English only. Hindi support planned for the AIS phase.
- Not a substitute for legal review. Citations are paragraph-precise but always verify against the official source.
- No real-time regulatory updates — the knowledge base is a dated snapshot.
- Solo-developed during nights and weekends; expect bugs and incomplete coverage in pre-alpha.

## Roadmap

- V1: FMVSS Part 571 (occupant crash protection) — in progress
- V2: Other FMVSS parts + NHTSA Recall Intelligence
- V3: AIS standards + Bharat Stage emission norms
- V4: UNECE WP.29 regulations
- Eventual: paid tier for individual consultants

## Why this exists

Automotive compliance engineers waste hours every week searching regulatory PDFs by Ctrl+F. Generic AI tools hallucinate critical clauses. Enterprise compliance software costs $50k/year and requires procurement approval. This tool sits in the gap.

Built by an automotive engineer (B.E. Mechanical, M.E. Automotive) who got tired of watching colleagues do this work the slow way.

## License

MIT — see [LICENSE](LICENSE).

## Author

Priyanka Yerolkar · [LinkedIn](https://linkedin.com/in/priyankayerolkar) · [GitHub](https://github.com/priyankayerolkar)