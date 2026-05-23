from typing import Protocol


class LLMClient(Protocol):
    """Vendor-agnostic chat client. Implementations live one file away."""

    model: str

    def generate(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        """Return the model's text answer for a system + user prompt."""
        ...
