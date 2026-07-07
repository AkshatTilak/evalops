"""Configuration loading and validation module using pydantic-settings."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings class.

    Loads configurations from environment variables or a local .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App Settings
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    # Database Settings
    database_url: str = Field(..., alias="DATABASE_URL")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")

    # API Keys
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")

    # Observability & Tracing (LangSmith / Phoenix)
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )
    langsmith_api_key: str = Field(..., alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="evalops-harness", alias="LANGSMITH_PROJECT")


@lru_cache
def get_settings() -> Settings:
    """Helper function to fetch cached Settings instance.

    Returns:
        An instance of Settings with parsed and validated parameters.
    """
    return Settings()
