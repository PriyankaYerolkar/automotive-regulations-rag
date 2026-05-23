import logging
from dataclasses import dataclass

from .config import settings
from .llm.base import LLMClient
from .llm.factory import build_llm

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is not legal advice. Verify against the official regulation "
    "before any compliance decision."
)

_SYSTEM = (
    "You answer questions about US automotive safety regulations using ONLY "
    "the provided context chunks. After every sentence that makes a factual "
    "claim, copy the exact [Source: ...] tag of the chunk that supports it, "
    "verbatim. If the context does not support an answer, reply exactly: "
    "'I could not find this in the retrieved documents.'"
)


@dataclass(frozen=True)
class Chunk:
    """A retrieved chunk plus the metadata needed to render a Skill 3 citation."""

    text: str
    regulation: str
    section: str
    subsection: str
    page: int
    effective_date: str


def render_citation(chunk: Chunk) -> str:
    """Build the trusted, code-controlled Skill 3 citation tag."""
    return (
        f"[Source: {chunk.regulation} §{chunk.section} {chunk.subsection}, "
        f"p.{chunk.page} — effective {chunk.effective_date}]"
    )


def _build_context(chunks: list[Chunk]) -> str:
    """Pair each chunk's text with its pre-rendered citation tag for the prompt."""
    return "\n\n".join(f"{c.text}\n{render_citation(c)}" for c in chunks)


def generate_answer(question: str, chunks: list[Chunk], *, llm: LLMClient | None = None) -> str:
    """Generate a cited compliance answer with the mandatory disclaimer appended."""
    if not chunks:
        logger.warning("generate_answer called with no chunks for: %s", question)
        return "\n\n".join(["I could not find this in the retrieved documents.", DISCLAIMER])

    client = llm or build_llm(settings.llm_model)  # eval injects the arm under test
    user = f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}"
    answer = client.generate(
        _SYSTEM, user, temperature=settings.temperature, max_tokens=settings.max_tokens
    )
    return f"{answer}\n\n{DISCLAIMER}"
