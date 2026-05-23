import logging
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AnthropicClient:
    """LLMClient backed by the Anthropic Messages API."""

    def __init__(self, model: str) -> None:
        self._client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )  # reads ANTHROPIC_API_KEY from env
        self.model = model

    def generate(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        """Call the Messages API and concatenate returned text blocks."""
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIError:
            logger.exception("Anthropic generation failed for model=%s", self.model)
            raise
        return "".join(block.text for block in resp.content if block.type == "text")
