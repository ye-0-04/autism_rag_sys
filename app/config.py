from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    LLM_BACKEND: Literal["local_vllm", "ollama", "openai", "anthropic"] = "ollama"

    VLLM_BASE_URL: str = "http://vllm:8000/v1"
    VLLM_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.3"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "qwen3.5:0.8b"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL_NAME: str = "claude-3-haiku-20240307"

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION_NAME: str = "nutrition_knowledge"

    API_SECRET_KEY: str = "change-this-in-production"
    RATE_LIMIT_PER_MINUTE: int = 10

    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def get_llm_provider():
    from app.llm.local_vllm import LocalvLLMProvider
    from app.llm.ollama import OllamaProvider
    from app.llm.openai_provider import OpenAIProvider
    from app.llm.anthropic_provider import AnthropicProvider

    providers = {
        "local_vllm": LocalvLLMProvider,
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    provider_class = providers.get(settings.LLM_BACKEND)
    if not provider_class:
        raise ValueError(f"Unknown LLM_BACKEND: {settings.LLM_BACKEND}")

    return provider_class()
