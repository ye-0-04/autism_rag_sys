import anthropic
from app.llm.base import LLMProvider, LLMResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model_name = settings.ANTHROPIC_MODEL_NAME

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    async def health_check(self) -> bool:
        try:
            await self.client.messages.create(
                model=self.model_name,
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
