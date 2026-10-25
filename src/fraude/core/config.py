"""Configuración central del proyecto."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
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


class QualityConfig(BaseModel):
    """Umbrales de la puerta de calidad (``configs/base.yaml`` -> ``quality``)."""

    model_config = ConfigDict(extra="forbid")

    min_side_px: int = Field(default=1000, gt=0)
    blur_threshold: float = Field(default=100.0, ge=0.0)
    max_skew_deg: float = Field(default=2.0, ge=0.0)
    min_contrast: float = Field(default=20.0, ge=0.0)
    overexposed_mean: float = Field(default=252.0, ge=0.0, le=255.0)
    underexposed_mean: float = Field(default=30.0, ge=0.0, le=255.0)


class ProjectConfig(BaseModel):
    """Configuración del pipeline leída desde YAML."""

    model_config = ConfigDict(extra="ignore")

    quality: QualityConfig = Field(default_factory=QualityConfig)


@lru_cache(maxsize=8)
def load_config(path: Path = Path("configs/base.yaml")) -> ProjectConfig:
    """Carga la configuración YAML; si falta el archivo, usa los valores por defecto."""
    if not path.exists():
        return ProjectConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ProjectConfig.model_validate(raw)


settings = Settings()
