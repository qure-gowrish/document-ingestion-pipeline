from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # LLM Provider — never hardcode model names, always from env
    llm_provider: Literal["openai", "anthropic", "portkey"] = "openai"
    llm_model_name: str = "gpt-4o-mini"
    llm_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = ""
    portkey_api_key: str = ""

    # Embedding — decoupled from LLM provider so Portkey LLM + direct OpenAI
    # embeddings can coexist without needing a second Portkey virtual key
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["openai", "portkey"] = "openai"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunking_strategy: Literal["fixed", "recursive", "sentence"] = "recursive"

    # Chroma
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "poc_documents"

    # LLM call limits (safety guards from architecture rules)
    llm_max_tokens: int = 500
    llm_temperature: float = 0.3
    llm_timeout: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    return Settings()
