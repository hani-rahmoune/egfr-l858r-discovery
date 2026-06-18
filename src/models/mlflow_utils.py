"""
MLflow experiment tracking helpers.

One experiment ("EGFR_QSAR_benchmark") holds all training runs so the
QSAR / GNN comparison lives in one place.

Usage
-----
    from src.models.mlflow_utils import start_run, log_seed_summary

    with start_run(task="general", model="gin", seed=42) as run:
        mlflow.log_params({"hidden_channels": 128, ...})
        mlflow.log_metrics({"val_rmse": 0.95, "test_rmse": 1.02, ...})
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "EGFR_QSAR_benchmark"


def _setup_mlflow():
    """Import mlflow and configure the tracking URI once."""
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    # Resolve relative URIs so they land in the project root, not cwd.
    if not uri.startswith(("http://", "https://", "databricks")):
        uri = str(get_project_root() / uri)
    mlflow.set_tracking_uri(uri)
    return mlflow


def get_or_create_experiment() -> str:
    """Return the experiment_id for EGFR_QSAR_benchmark, creating it if absent."""
    mlflow = _setup_mlflow()
    # mlflow.set_experiment() creates the experiment if absent and returns it.
    exp = mlflow.set_experiment(EXPERIMENT_NAME)
    return exp.experiment_id


@contextlib.contextmanager
def start_run(
    task: str, model: str, seed: int | None = None, run_name: str | None = None
):
    """
    Context manager that starts an MLflow run under EGFR_QSAR_benchmark.

    Adds standard tags automatically:
        task   — "general" or "wt_proxy"
        model  — "random_forest", "xgboost", "lightgbm", "gin", …
        seed   — integer or "summary"
    """
    mlflow = _setup_mlflow()
    exp_id = get_or_create_experiment()

    if run_name is None:
        seed_str = str(seed) if seed is not None else "summary"
        run_name = f"{task}__{model}__seed{seed_str}"

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.set_tags(
            {
                "task": task,
                "model": model,
                "seed": str(seed) if seed is not None else "summary",
            }
        )
        yield run


def log_seed_summary(
    task: str,
    model: str,
    per_seed: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> None:
    """
    Log a summary run with mean ± std aggregated over per-seed metric dicts.

    Logged metrics follow the pattern  {metric}_mean  and  {metric}_std.
    The full per-seed table is also logged as an artifact (JSON).
    """
    import json
    import tempfile
    from pathlib import Path

    import numpy as np

    mlflow = _setup_mlflow()

    # Collect numeric metric keys present in all dicts
    metric_keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float))]

    summary: dict[str, float] = {}
    for k in metric_keys:
        vals = [m[k] for m in per_seed if isinstance(m.get(k), (int, float))]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))

    with start_run(task=task, model=model, seed=None):
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(summary)

        # Log per-seed table as a JSON artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=f"{task}_{model}_seeds_"
        ) as f:
            json.dump(per_seed, f, indent=2)
            tmp = Path(f.name)
        try:
            mlflow.log_artifact(str(tmp), artifact_path="per_seed")
        finally:
            tmp.unlink(missing_ok=True)

    logger.info(
        f"Logged summary run: {task}/{model}  "
        + "  ".join(f"{k}={v:.3f}" for k, v in summary.items() if k.endswith("_mean"))
    )
