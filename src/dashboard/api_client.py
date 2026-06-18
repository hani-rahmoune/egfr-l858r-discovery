"""
Scoring client for the dashboard (Phase 25).

Prefers the running FastAPI service (Phase 24) at `base_url`; if it is not up,
falls back to scoring locally through the same `ModelRegistry` the API uses.
Either way the dashboard gets the identical fast-screen dict shape.

Pure decision helpers (`api_available`, `predict_via_api`, `predict_via_registry`)
are split out so they can be unit-tested without a network or a real registry.
"""

from __future__ import annotations

import os
from typing import Any

import requests

# In Docker the dashboard reaches the API by service name; override with env var.
DEFAULT_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def api_available(base_url: str = DEFAULT_BASE_URL, timeout: float = 0.5) -> bool:
    """True iff GET {base_url}/health returns 200 with status 'ok'."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def predict_via_api(
    base_url: str, smiles: str, timeout: float = 10.0
) -> dict[str, Any]:
    """Call POST /predict. Raises requests.HTTPError on 4xx/5xx."""
    r = requests.post(
        f"{base_url.rstrip('/')}/predict", json={"smiles": smiles}, timeout=timeout
    )
    if r.status_code >= 400:
        # surface the structured {error, detail} body as an exception payload
        raise requests.HTTPError(r.json())
    return r.json()


def batch_predict_via_api(
    base_url: str, smiles_list: list[str], timeout: float = 60.0
) -> dict[str, Any]:
    """Call POST /batch_predict."""
    r = requests.post(
        f"{base_url.rstrip('/')}/batch_predict",
        json={"smiles": smiles_list},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def predict_via_registry(registry: Any, smiles: str) -> dict[str, Any]:
    """Score locally via a loaded ModelRegistry (same dict shape as the API)."""
    return registry.score(smiles)


def model_info_via_api(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    r = requests.get(f"{base_url.rstrip('/')}/model-info", timeout=timeout)
    r.raise_for_status()
    return r.json()
