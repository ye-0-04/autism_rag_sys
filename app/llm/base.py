from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass
