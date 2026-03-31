from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_ENV: str = "development"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    STRICT_ENV_VALIDATION: bool = True

    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8081"
    DEFAULT_WARDROBE_FETCH_LIMIT: int = 1000
    APPWRITE_PAGE_SIZE: int = 100
    MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024

    APPWRITE_ENDPOINT: Optional[str] = None
    EXPO_PUBLIC_APPWRITE_ENDPOINT: Optional[str] = None
    APPWRITE_PROJECT_ID: Optional[str] = None
    EXPO_PUBLIC_APPWRITE_PROJECT_ID: Optional[str] = None
    APPWRITE_DATABASE_ID: Optional[str] = None
    EXPO_PUBLIC_APPWRITE_DATABASE_ID: Optional[str] = None
    APPWRITE_API_KEY: Optional[str] = None
    EXPO_PUBLIC_APPWRITE_API_KEY: Optional[str] = None
    APPWRITE_KEY: Optional[str] = None

    @property
    def is_production(self) -> bool:
        env = (self.ENVIRONMENT or self.APP_ENV or "").strip().lower()
        return env in {"prod", "production"}

    @property
    def docs_enabled(self) -> bool:
        return not self.is_production

    @property
    def cors_origins(self) -> List[str]:
        origins = [o.strip() for o in (self.ALLOWED_ORIGINS or "").split(",") if o.strip()]
        return origins

    @model_validator(mode="after")
    def _validate_required_config(self) -> "AppSettings":
        if not self.STRICT_ENV_VALIDATION:
            return self

        missing = []
        if not (self.APPWRITE_ENDPOINT or self.EXPO_PUBLIC_APPWRITE_ENDPOINT):
            missing.append("APPWRITE_ENDPOINT")
        if not (self.APPWRITE_PROJECT_ID or self.EXPO_PUBLIC_APPWRITE_PROJECT_ID):
            missing.append("APPWRITE_PROJECT_ID")
        if not (self.APPWRITE_DATABASE_ID or self.EXPO_PUBLIC_APPWRITE_DATABASE_ID):
            missing.append("APPWRITE_DATABASE_ID")
        if not (self.APPWRITE_API_KEY or self.EXPO_PUBLIC_APPWRITE_API_KEY or self.APPWRITE_KEY):
            missing.append("APPWRITE_API_KEY")

        if "*" in self.cors_origins:
            missing.append("ALLOWED_ORIGINS must not include '*' in STRICT_ENV_VALIDATION mode")
        if not self.cors_origins:
            missing.append("ALLOWED_ORIGINS")

        if missing:
            raise ValueError("Missing/invalid required configuration: " + ", ".join(missing))
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def configure_logging() -> None:
    level_name = str(os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
