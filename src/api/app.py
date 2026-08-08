"""
Bhashini NHLT-compatible FastAPI Application

Implements the cloud deployment layer for the Rajasthani Dialect AI.
Enforces:
- API key authentication
- DPDP Act 2023 compliance (India-first data localization, 30-day log retention)
- TLS/HTTPS encrypted transit
- Structured JSON logging
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from src.api.routes.translate import router as translate_router
from src.api.routes.health import router as health_router
from src.api.middleware.dpdp import DPDPComplianceMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup, release on shutdown."""
    logger.info("Starting Rajasthani Dialect AI API Server")
    # Model loading would happen here in production
    # app.state.asr_model = load_asr(...)
    # app.state.mt_model = load_mt(...)
    # app.state.tts_model = load_tts(...)
    yield
    logger.info("Shutting down API Server")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    app = FastAPI(
        title="Rajasthani Dialect AI — Bhashini NHLT API",
        description=(
            "Cascaded S2ST translation API for Marwari, Mewari, Dhundhari, "
            "Hadoti, Mewati, and Bagri dialects. DPDP Act 2023 compliant."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # DPDP compliance middleware (logging, data localization enforcement)
    app.add_middleware(DPDPComplianceMiddleware)

    # Route registration
    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(translate_router, prefix="/api/v1", tags=["Translation"])

    return app


# Default app instance for `uvicorn src.api.app:app`
app = create_app()
