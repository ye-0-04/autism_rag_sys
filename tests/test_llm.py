import pytest
import os

os.environ["LLM_BACKEND"] = "ollama"

from app.config import get_llm_provider
from app.llm.base import LLMResponse


@pytest.mark.asyncio
async def test_provider_factory_returns_correct_type():
    os.environ["LLM_BACKEND"] = "ollama"
    from app.config import get_settings

    get_settings.cache_clear()
    provider = get_llm_provider()
    from app.llm.ollama import OllamaProvider

    assert isinstance(provider, OllamaProvider)


@pytest.mark.asyncio
async def test_ollama_health_check():
    from app.llm.ollama import OllamaProvider

    provider = OllamaProvider()
    result = await provider.health_check()
    assert result is True, "Ollama health check failed — is it running?"


@pytest.mark.asyncio
async def test_ollama_generate_returns_llmresponse():
    from app.llm.ollama import OllamaProvider

    provider = OllamaProvider()
    response = await provider.generate(
        user_prompt="What is vitamin D?",
        system_prompt="You are a nutrition expert. Be concise.",
        max_tokens=100,
    )
    assert isinstance(response, LLMResponse)
    assert response.content is not None
    assert len(response.content) > 0
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0


@pytest.mark.asyncio
async def test_provider_interface_is_consistent():
    from app.llm.local_vllm import LocalvLLMProvider
    from app.llm.ollama import OllamaProvider
    from app.llm.openai_provider import OpenAIProvider
    from app.llm.anthropic_provider import AnthropicProvider

    for ProviderClass in [
        LocalvLLMProvider,
        OllamaProvider,
        OpenAIProvider,
        AnthropicProvider,
    ]:
        provider = ProviderClass()
        assert hasattr(provider, "generate")
        assert hasattr(provider, "health_check")
        assert callable(provider.generate)
        assert callable(provider.health_check)
