"""
LLM provider factory for the document ingestion POC.

Returns LangChain BaseChatModel instances so the enrichment layer can use
LCEL (LangChain Expression Language) chains:

    chain = prompt | llm | StrOutputParser()

Three providers are supported:
  - openai    : ChatOpenAI  (default)
  - anthropic : ChatAnthropic
  - portkey   : ChatOpenAI routed through the Portkey gateway

Architecture rules applied:
  - Never hardcode model names — always injected from config / env vars.
  - Chains must be stateless — each call to get_provider() returns a fresh model.
  - Add timeout guard on tool execution — timeout_seconds is passed to every model.
  - Validate tool input schema strictly — done inside the LCEL chain in enricher.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel


def get_provider(
    provider_type: str,
    api_key: str,
    model_name: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    timeout_seconds: int = 30,
) -> BaseChatModel:
    """
    Factory — returns a LangChain BaseChatModel for the given provider.
    Switching providers requires only a change to LLM_PROVIDER in .env.
    """
    common: dict[str, Any] = dict(
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout_seconds,
    )

    if provider_type == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(model=model_name, api_key=api_key, **common)

    elif provider_type == "anthropic":
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
        return ChatAnthropic(model=model_name, api_key=api_key, **common)

    elif provider_type == "portkey":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders  # noqa: PLC0415
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=PORTKEY_GATEWAY_URL,
            default_headers=createHeaders(api_key=api_key),
            **common,
        )

    else:
        raise ValueError(
            f"Unsupported provider '{provider_type}'. Choose: openai | anthropic | portkey"
        )
