from .anthropic_client import AnthropicClient
from .base import LLMClient


def build_llm(model: str) -> LLMClient:
    """Map a model string to a provider client. Add vendors here, nowhere else."""
    if model.startswith("claude-"):
        return AnthropicClient(model)
    # if model.startswith("gpt-"):
    #     return OpenAIClient(model)   # Skill 1 comparison arm — one line when needed
    raise ValueError(f"No provider registered for model: {model!r}")