from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Model is a string so the LLM is swappable."""

    llm_model: str = "claude-haiku-4-5-20251001"  # quality arm: claude-sonnet-4-6
    embedding_model: str = "text-embedding-3-small"  # locked in Phase 2
    temperature: float = 0.1  # Skill 1: factual answers
    max_tokens: int = 1024
    top_k: int = 5  # Skill 1
    mmr_lambda: float = 0.7  # raised 0.5→0.7 (2026-05-28): regulatory Q&A favors
                              # relevance over diversity; S6.2 deferred to rank 5
                              # at 0.5, rank 4 at 0.7. Re-tune after 30-question eval.


settings = Settings()
