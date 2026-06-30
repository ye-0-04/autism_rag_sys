import httpx
from app.llm.base import LLMProvider, LLMResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model_name = settings.OLLAMA_MODEL_NAME
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
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }

        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        content = data["message"].get("content", "") or data["message"].get(
            "thinking", ""
        )

        return LLMResponse(
            content=content,
            model=data["model"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )

    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
