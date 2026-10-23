"""Configuración central del proyecto."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRAUDE_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    mlflow_tracking_uri: str = "file:///app/artifacts/mlruns"
    ollama_host: str = "http://localhost:11434"
    seed: int = 42


settings = Settings()
