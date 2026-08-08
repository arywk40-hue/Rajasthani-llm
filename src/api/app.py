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
    
    # Lazy imports to avoid heavy loading if models are not needed
    from src.asr.model import WhisperASR
    from src.mt.model import IndicTrans2MT
    from src.tts.fastpitch import IndicTTS
    
    # In a real production deployment, you'd likely configure which models to load
    # based on environment variables to save memory (e.g., LOAD_ASR=true)
    try:
        logger.info("Initializing MT Model...")
        app.state.mt_model = IndicTrans2MT()
    except Exception as e:
        logger.warning(f"Could not load MT model: {e}")
        app.state.mt_model = None

    try:
        logger.info("Initializing ASR Model...")
        app.state.asr_model = WhisperASR()
    except Exception as e:
        logger.warning(f"Could not load ASR model: {e}")
        app.state.asr_model = None

    try:
        logger.info("Initializing TTS Model...")
        app.state.tts_model = IndicTTS()
    except Exception as e:
        logger.warning(f"Could not load TTS model: {e}")
        app.state.tts_model = None

    yield
    
    logger.info("Shutting down API Server")
    # Release resources
    app.state.mt_model = None
    app.state.asr_model = None
    app.state.tts_model = None

def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

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

    # Serve static directory & UI
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")

        @app.get("/", include_in_schema=False)
        async def root():
            return FileResponse("static/index.html")

    return app


# Default app instance for `uvicorn src.api.app:app`
app = create_app()
