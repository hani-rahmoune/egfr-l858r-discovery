"""Load YAML configs and environment variables; resolve all paths against project root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Two parents up from src/utils/config.py reaches the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_paths() -> dict[str, Any]:
    return _load_yaml(PROJECT_ROOT / "config" / "paths.yaml")


def load_model_config() -> dict[str, Any]:
    return _load_yaml(PROJECT_ROOT / "config" / "model_config.yaml")


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_path(key_path: str) -> Path:
    paths = load_paths()
    keys = key_path.split(".")
    value = paths
    for k in keys:
        value = value[k]
    return PROJECT_ROOT / value


def get_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def is_cloud_mode() -> bool:
    return get_env("USE_GCS", "false").lower() == "true"


def get_seed() -> int:
    return int(get_env("RANDOM_SEED", "42"))
