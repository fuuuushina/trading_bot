"""
config/loader.py
Centralised configuration loader — reads YAML files and exposes typed config.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    return _load_yaml("settings.yaml")


@lru_cache(maxsize=1)
def get_risk_config() -> dict[str, Any]:
    return _load_yaml("risk.yaml")


@lru_cache(maxsize=1)
def get_strategy_config() -> dict[str, Any]:
    return _load_yaml("strategies.yaml")


def get_env(key: str, default: str | None = None) -> str:
    """Read environment variable, raise if missing and no default."""
    value = os.environ.get(key, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Check your .env file or shell environment."
        )
    return value


def is_live_enabled() -> bool:
    settings = get_settings()
    return (
        settings["system"]["mode"] == "live"
        and settings["system"]["live_enabled"] is True
    )
