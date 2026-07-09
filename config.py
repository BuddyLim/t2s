"""Application configuration, loaded from environment / .env file.

All settings can be overridden via environment variables (or a local .env).
Model ids and the optional base_url make it easy to swap Claude models or
point at an OpenAI/Anthropic-compatible gateway without touching code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / a local .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Credentials -------------------------------------------------------
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key. Required unless using a custom gateway.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional custom endpoint (gateway / local) for the provider.",
    )

    # --- Models ------------------------------------------------------------
    sql_model: str = Field(
        default="claude-haiku-4-5",
        description="Fast/cheap model used to generate SQL.",
    )
    answer_model: str = Field(
        default="claude-sonnet-5",
        description="Stronger model used to phrase the natural-language answer.",
    )

    # --- Source documents --------------------------------------------------
    excel_path: Path = Field(
        default=Path("data.xlsx"),
        description="Path to the Excel workbook used as the database.",
    )
    dict_path: Path | None = Field(
        default=None,
        description=(
            "Optional path to the Word document used as the data dictionary. "
            "If unset or missing, the tool runs Excel-only."
        ),
    )


def load_settings() -> Settings:
    """Load settings from environment / .env."""
    return Settings()


def build_model(settings: Settings, model_name: str) -> AnthropicModel:
    """Construct an Anthropic-backed model, honouring an optional base_url.

    The same provider config drives both the SQL and answer models, so callers
    just pass the model id (settings.sql_model or settings.answer_model).
    """
    provider = AnthropicProvider(
        api_key=settings.anthropic_api_key or None,
        base_url=settings.base_url,
    )
    return AnthropicModel(model_name, provider=provider)
