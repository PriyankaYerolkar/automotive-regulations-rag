import json
import logging
import re

from ..generate import Chunk, render_citation
from ..llm.base import LLMClient

logger = logging.getLogger(__name__)

REFUSAL = "i could not find this in the retrieved documents"
_TAG_RE = re.compile(r"\[Source:[^\]]+\]")
_JUDGE_SYSTEM = (
    "You are a strict grader. Given CONTEXT and an ANSWER, decide whether EVERY "
    "factual claim in the ANSWER is directly supported by the CONTEXT. Respond with "
    'ONLY JSON: {"supported": true|false, "unsupported": ["..."]}. No prose, no fences.'
)


def emitted_tags(answer: str) -> list[str]:
    """Every [Source: ...] tag the model wrote, in order."""
    return _TAG_RE.findall(answer)


def _context_tags(chunks: list[Chunk]) -> set[str]:
    """Valid citation tags for the chunks the model was given."""
    return {render_citation(c) for c in chunks}


def citation_score(answer: str, chunks: list[Chunk]) -> float:
    """Answerable item: 1.0 if it cites and every tag exists in context, else 0.0.

    Because citations are rendered in code and copied verbatim, a tag that is not
    in context is a fabrication — the dominant citation failure mode.
    """
    tags = emitted_tags(answer)
    if not tags:
        return 0.0
    return 1.0 if all(t in _context_tags(chunks) for t in tags) else 0.0


def bait_hallucinated(answer: str, chunks: list[Chunk]) -> bool:
    """True if a bait item was mishandled: no refusal, or a fabricated citation."""
    refused = REFUSAL in answer.lower()
    fabricated = any(t not in _context_tags(chunks) for t in emitted_tags(answer))
    return fabricated or not refused


def faithfulness(answer: str, chunks: list[Chunk], judge: LLMClient) -> float:
    """LLM-judged: 1.0 if all claims are supported by context, else 0.0."""
    context = "\n\n".join(c.text for c in chunks)
    raw = judge.generate(
        _JUDGE_SYSTEM, f"CONTEXT:\n{context}\n\nANSWER:\n{answer}",
        temperature=0.0, max_tokens=512,
    ).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # Judge sometimes adds reasoning prose after the JSON — extract just the object
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        logger.warning("Judge returned no JSON object; scoring 0: %s", raw[:200])
        return 0.0
    try:
        verdict = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("Judge JSON parse failed; scoring 0: %s", match.group()[:200])
        return 0.0
    return 1.0 if verdict.get("supported") else 0.0