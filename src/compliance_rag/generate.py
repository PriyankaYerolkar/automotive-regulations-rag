import logging
import re
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

# Bare section numbers (e.g. "S6", "S15.3") carry no qualifier, so they are not
# worth prepending to context — only real descriptive titles are.
_BARE_SNUM = re.compile(r"S[0-9.]+")

@dataclass(frozen=True)
class Chunk:
    """A retrieved chunk plus the metadata needed to render a Skill 3 citation."""

    text: str
    regulation: str
    section: str
    subsection: str
    page: int
    effective_date: str
    # Parent section heading (e.g. "Injury criteria for the 5th percentile adult
    # female dummy"). Carries dummy/vehicle-class qualifiers the paragraph text
    # omits. Defaults to "" so existing callers keep working.
    parent_heading: str = ""


def render_citation(chunk: Chunk) -> str:
    """Build the trusted, code-controlled Skill 3 citation tag."""
    return (
        f"[Source: {chunk.regulation} §{chunk.section} {chunk.subsection}, "
        f"p.{chunk.page} — effective {chunk.effective_date}]"
    )


def _build_context(chunks: list[Chunk]) -> str:
    """Pair each chunk's parent heading + text with its pre-rendered citation tag.

    The parent section heading carries qualifiers (dummy size, vehicle class) that
    the bare paragraph omits. Including it lets the model answer questions that hinge
    on those qualifiers faithfully, instead of refusing because the paragraph never
    restates them. The heading is context only — it is NOT part of the citation tag.
    """
    blocks: list[str] = []
    for c in chunks:
        head_text = c.parent_heading.strip()
        show = bool(head_text) and not _BARE_SNUM.fullmatch(head_text)
        head = f"[{head_text}]\n" if show else ""
        blocks.append(f"{head}{c.text}\n{render_citation(c)}")
    return "\n\n".join(blocks)


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