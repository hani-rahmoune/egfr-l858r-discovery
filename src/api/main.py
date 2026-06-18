"""
FastAPI application for the EGFR L858R fast-screen service (Phase 24).

Serves PRECOMPUTED models only. At startup (lifespan) it loads every artifact
ONCE — backbone activity (Model 1), WT-proxy (Model 2), the ADMET filter, the
covalent detector, and the applicability domain — into `app.state.registry`.
No training, docking, or ESM-2 runs at request time.

Run locally:
    PYTHONPATH=. .venv/Scripts/python.exe -m uvicorn src.api.main:app --reload
    # docs at http://127.0.0.1:8000/docs

`create_app()` builds the app; `app` is the module-level instance uvicorn serves.
Tests import `create_app()` and override the `get_registry` dependency with a
mock, so no real artifacts are needed to exercise the endpoints.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import router
from src.api.services import SERVICE_VERSION, ModelRegistry
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DESCRIPTION = (
    "Fast screening API for EGFR L858R NSCLC drug discovery. Serves precomputed "
    "models only: backbone activity, WT-proxy, ADMET, covalent detector, and "
    "applicability domain. Activity numbers are EXPLORATORY. Docking-based "
    "selectivity is NOT computed at request time — see GET /model-info."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts once at startup; clear them on shutdown."""
    try:
        logger.info("Loading model artifacts ...")
        app.state.registry = ModelRegistry.load()
        logger.info("Model artifacts loaded; service ready.")
    except Exception as exc:  # pragma: no cover - startup failure path
        logger.error(f"Failed to load model artifacts: {exc}")
        app.state.registry = None
    yield
    app.state.registry = None


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    """
    Build the FastAPI app.

    If `registry` is provided (tests), it is used directly and the loading
    lifespan is skipped. Otherwise artifacts are loaded at startup.
    """
    app = FastAPI(
        title="EGFR L858R Fast-Screen API",
        version=SERVICE_VERSION,
        description=_DESCRIPTION,
        lifespan=None if registry is not None else lifespan,
    )

    if registry is not None:
        app.state.registry = registry

    app.include_router(router)
    _install_error_handlers(app)
    return app


def _install_error_handlers(app: FastAPI) -> None:
    """Return structured {error, detail} bodies for all 4xx/5xx responses."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exc_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            body = detail
        else:
            body = {"error": f"http_{exc.status_code}", "detail": str(detail)}
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_exc_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {
                    "error": "validation_error",
                    "detail": "Request body failed validation.",
                    "errors": exc.errors(),
                }
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exc_handler(
        request: Request, exc: Exception
    ):  # pragma: no cover
        logger.error(f"Unhandled error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "An unexpected error occurred.",
            },
        )


app = create_app()
