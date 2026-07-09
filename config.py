"""Application configuration, loaded from environment / .env file.

All settings can be overridden via environment variables (or a local .env).
Model ids and the optional base_url make it easy to swap Claude models or
point at an Anthropic-compatible gateway without touching code.
"""

from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from pydantic import Field, SecretStr
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


def build_model(settings: Settings, model_name: str) -> ChatAnthropic:
    """Construct an Anthropic-backed chat model, honouring an optional base_url.

    The same provider config drives both the SQL and answer models, so callers
    just pass the model id (settings.sql_model or settings.answer_model). A blank
    api_key is passed as None so ChatAnthropic falls back to ANTHROPIC_API_KEY;
    a None base_url uses Anthropic's default endpoint.
    """
    api_key = SecretStr(settings.anthropic_api_key) if settings.anthropic_api_key else None
    return ChatAnthropic(
        model_name=model_name,
        api_key=api_key,  # pyright: ignore[reportArgumentType]  # None => ANTHROPIC_API_KEY
        base_url=settings.base_url,
        timeout=None,
        stop=None,
    )
