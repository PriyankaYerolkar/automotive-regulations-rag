import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..generate import Chunk, generate_answer
from ..llm.factory import build_llm
from .scoring import bait_hallucinated, citation_score, faithfulness

_CHUNK_FIELDS = {"text", "regulation", "section", "subsection", "page", "effective_date", "parent_heading"}

logger = logging.getLogger(__name__)

CITATION_THRESHOLD = 0.95
FAITHFULNESS_THRESHOLD = 0.90


@dataclass(frozen=True)
class EvalItem:
    id: str
    category: str
    question: str
    chunks: list[Chunk]
    is_bait: bool = False


@dataclass
class ModelReport:
    model: str
    n_answerable: int
    citation_accuracy: float
    faithfulness: float
    bait_failures: list[str] = field(default_factory=list)
    per_category: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """All Skill 4 generation thresholds met (retrieval metrics excluded)."""
        return (
            self.citation_accuracy >= CITATION_THRESHOLD
            and self.faithfulness >= FAITHFULNESS_THRESHOLD
            and not self.bait_failures
        )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def load_eval_set(path: Path) -> list[EvalItem]:
    """Parse the version-controlled eval set JSON into typed items."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalItem(
            r["id"],
            r["category"],
            r["question"],
            [Chunk(**{k: v for k, v in c.items() if k in _CHUNK_FIELDS}) for c in r["chunks"]],
            r.get("is_bait", False),
        )
        for r in raw
    ]


def run_model(model: str, items: list[EvalItem], judge_model: str) -> ModelReport:
    """Run one model through the same generate path over the whole eval set."""
    llm = build_llm(model)
    judge = build_llm(judge_model)
    cit: list[float] = []
    faith: list[float] = []
    bait_failures: list[str] = []
    cat: dict[str, list[float]] = {}
    for item in items:
        answer = generate_answer(item.question, item.chunks, llm=llm)
        if item.is_bait:
            if bait_hallucinated(answer, item.chunks):
                bait_failures.append(item.id)
            continue
        score = citation_score(answer, item.chunks)
        cit.append(score)
        faith.append(faithfulness(answer, item.chunks, judge))
        cat.setdefault(item.category, []).append(score)
    logger.info("Scored %s: %d answerable, %d bait failures", model, len(cit), len(bait_failures))
    return ModelReport(
        model=model,
        n_answerable=len(cit),
        citation_accuracy=_mean(cit),
        faithfulness=_mean(faith),
        bait_failures=bait_failures,
        per_category={k: _mean(v) for k, v in cat.items()},
    )
