"""RegCite - cited Q&A demo for FMVSS Section 571.208.

A thin Gradio wrapper over the V1 RAG backend. It takes a question, runs
retrieve -> MMR rerank -> generate (Skill 1), and shows:
  - the cited answer (each factual sentence followed by a [Source: ...] tag),
  - the mandatory "not legal advice" disclaimer (always visible),
  - the retrieved source chunks in a side panel.

The Space provides OPENAI_API_KEY and ANTHROPIC_API_KEY as injected env vars
(set them under Settings -> Variables and secrets as *Secrets*). No .env needed.
"""

from __future__ import annotations

import logging
import os
import re

import gradio as gr

from src.compliance_rag.generate import render_citation
from src.compliance_rag.pipeline import QueryResult, answer_with_sources

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("regcite.app")

TITLE = "RegCite"
SUBTITLE = (
    "Cited Q&A for automotive safety regulations &middot; "
    "**V1: US FMVSS \u00a7571.208** (occupant crash protection)"
)

# A persistent banner so the disclaimer is visible even before any answer.
# The generated answer also ends with this same disclaimer (from the backend).
# Rendered as inline-styled HTML so it does not depend on a global `css=` arg,
# which moved off the Blocks constructor in Gradio 6.
_ECFR_URL = (
    "https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/"
    "part-571/subpart-B/section-571.208"
)
DISCLAIMER_HTML = (
    '<div style="border:1px solid #d97706;border-left:4px solid #d97706;'
    'border-radius:8px;padding:10px 14px;background:rgba(217,119,6,0.10);">'
    "<strong>Not legal advice.</strong> This tool answers from a single public "
    "regulation (US FMVSS \u00a7571.208, occupant crash protection) and may be "
    "incomplete or wrong. Verify every clause against the official regulation on "
    f'<a href="{_ECFR_URL}" target="_blank" rel="noopener">eCFR</a> '
    "before any compliance decision."
    "</div>"
)

# Real questions grounded in Section 571.208 content, plus one out-of-scope item
# that demonstrates the honest refusal ("I could not find this...").
EXAMPLE_QUESTIONS = [
    "What is the HIC limit under FMVSS 571.208?",
    "What is the maximum femur load for the 5th percentile adult female dummy?",
    "What is the maximum chest acceleration permitted under FMVSS 571.208?",
    "What is the maximum HIC15 value permitted for the Hybrid III dummy?",
    "What does FMVSS 208 require for autonomous-vehicle sensors?",  # out of scope on purpose
]

_KEYS_MISSING_MSG = (
    "**Configuration error.** `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` are "
    "not set. On Hugging Face Spaces, add them under "
    "*Settings -> Variables and secrets* as **Secrets**, then restart the Space."
)
_GENERIC_ERROR_MSG = (
    "Something went wrong answering that question. Please try again in a moment."
)
_EMPTY_SOURCES_MSG = "_Ask a question to see the regulation paragraphs that were retrieved._"

# Make the code-rendered [Source: ...] tags stand out in the answer.
_SOURCE_TAG = re.compile(r"(\[Source:[^\]]*\])")


def _keys_present() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY"))


def _highlight_citations(answer: str) -> str:
    """Bold every [Source: ...] tag so citations are visually distinct."""
    return _SOURCE_TAG.sub(r"**\1**", answer)


def _format_sources(result: QueryResult) -> str:
    """Render retrieved chunks as a readable markdown list for the side panel."""
    if not result.chunks:
        return "_No source paragraphs were retrieved for this question._"

    blocks: list[str] = []
    for i, c in enumerate(result.chunks, start=1):
        heading = c.parent_heading.strip()
        heading_line = f"*{heading}*\n\n" if heading else ""
        snippet = c.text.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rstrip() + " \u2026"
        blocks.append(
            f"**{i}. `{render_citation(c)}`**\n\n{heading_line}{snippet}"
        )
    return "\n\n---\n\n".join(blocks)


def run_query(question: str) -> tuple[str, str]:
    """Answer one question; return (answer_markdown, sources_markdown)."""
    if not question or not question.strip():
        return "Enter a question about FMVSS \u00a7571.208 above.", _EMPTY_SOURCES_MSG

    if not _keys_present():
        return _KEYS_MISSING_MSG, _EMPTY_SOURCES_MSG

    try:
        result = answer_with_sources(question.strip())
    except Exception:  # noqa: BLE001 - demo surface: log detail, show generic msg
        logger.exception("query failed for: %s", question[:120])
        return _GENERIC_ERROR_MSG, _EMPTY_SOURCES_MSG

    return _highlight_citations(result.answer), _format_sources(result)


def _clear() -> tuple[str, str, str]:
    return "", "", _EMPTY_SOURCES_MSG


with gr.Blocks(title=f"{TITLE} - FMVSS 571.208 Q&A") as demo:
    gr.Markdown(f"# {TITLE}\n{SUBTITLE}")
    gr.HTML(DISCLAIMER_HTML)

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is the HIC limit under FMVSS 571.208?",
                lines=2,
                autofocus=True,
            )
            with gr.Row():
                submit = gr.Button("Ask", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUESTIONS],
                inputs=question,
                label="Try one of these (the last is intentionally out of scope)",
            )
            gr.Markdown("### Answer")
            answer_box = gr.Markdown(
                "Answers cite the supporting paragraph after each claim. "
                "Out-of-scope questions return *\u201cI could not find this in the "
                "retrieved documents.\u201d* rather than guessing.",
            )

        with gr.Column(scale=2):
            gr.Markdown("### Retrieved sources")
            sources_box = gr.Markdown(_EMPTY_SOURCES_MSG)

    gr.Markdown(
        "Scope: **FMVSS \u00a7571.208 only** (frontal occupant crash protection - "
        "airbags, belts, injury criteria). Public source: NHTSA / eCFR Title 49. "
        "Not affiliated with NHTSA. **Not legal advice.**"
    )

    submit.click(run_query, inputs=question, outputs=[answer_box, sources_box])
    question.submit(run_query, inputs=question, outputs=[answer_box, sources_box])
    clear.click(_clear, inputs=None, outputs=[question, answer_box, sources_box])


if __name__ == "__main__":
    demo.queue()  # serialize concurrent requests on the free tier
    demo.launch()