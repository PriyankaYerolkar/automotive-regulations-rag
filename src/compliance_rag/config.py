from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Model is a string so the LLM is swappable."""

    llm_model: str = "claude-sonnet-4-6"  # cheap arm in eval: claude-haiku-4-5-20251001
    embedding_model: str = "text-embedding-3-small"  # locked in Phase 2
    temperature: float = 0.1  # Skill 1: factual answers
    max_tokens: int = 1024
    top_k: int = 5  # Skill 1
    mmr_lambda: float = 0.7  # was 0.5; regulatory Q&A needs relevance > diversity


settings = Settings()
