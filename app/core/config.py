# Copyright 2024
# Directory: yt-rag/app/core/config.py

"""
Configuration management for the RAG application.
Handles environment variables and application settings.
"""

from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Supabase Configuration
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


    # AI Provider Configuration
    ai_provider: Literal["openai", "anthropic", "gemini"] = Field(
        default="openai",
        env="AI_PROVIDER"
    )


    # HuggingFace Configuration (Embedding)
    hf_token: str = Field(
        default="",
        env="HF_TOKEN"
    )


    # OpenRouter / OpenAI Compatible Configuration
    openrouter_api_key: str = Field(
        default="",
        env="OPENROUTER_API_KEY"
    )


    # OpenAI fields
    openai_api_key: str = Field(
        default="",
        env="OPENAI_API_KEY"
    )

    openai_chat_model: str = Field(
        default="meta-llama/llama-3.1-8b-instruct",
        env="OPENAI_CHAT_MODEL"
    )

    openai_embed_model: str = Field(
        default="text-embedding-3-small",
        env="OPENAI_EMBED_MODEL"
    )


    # Gemini Configuration
    gemini_api_key: str = Field(
        default="",
        env="GEMINI_API_KEY"
    )

    gemini_chat_model: str = Field(
        default="gemini-2.0-flash",
        env="GEMINI_CHAT_MODEL"
    )


    # Anthropic Configuration
    anthropic_api_key: str = Field(
        default="",
        env="ANTHROPIC_API_KEY"
    )

    anthropic_chat_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        env="ANTHROPIC_CHAT_MODEL"
    )


    # Application Configuration
    environment: str = Field(
        default="development",
        env="ENVIRONMENT"
    )

    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL"
    )


    # RAG Configuration
    default_top_k: int = Field(default=6)

    chunk_size: int = Field(
        default=400
    )

    chunk_overlap: int = Field(
        default=60
    )

    temperature: float = Field(
        default=0.1
    )

    embedding_dimensions: int = Field(
        default=384
    )


    class Config:
        env_file = ".env"
        case_sensitive = False



# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    return settings