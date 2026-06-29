import httpx
from app.llm.base import LLMProvider, LLMResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class LocalvLLMProvider(LLMProvider):
    def __init__(self):
        self.base_url = settings.VLLM_BASE_URL
        self.model_name = settings.VLLM_MODEL_NAME
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.info(f"Sending request to vLLM: {self.base_url}/chat/completions")

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
        )

    async def health_check(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.base_url.replace('/v1', '')}/health"
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"vLLM health check failed: {e}")
            return False
