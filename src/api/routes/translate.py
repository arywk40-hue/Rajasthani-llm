"""
Translation API Routes

Exposes the cascaded S2ST pipeline (ASR → MT → TTS) as REST endpoints,
mirroring the Bhashini NHLT API contract.
"""

import base64
import hmac
import io
import os

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

router = APIRouter()

# API key is loaded once at module import time from environment.
# Set via:  export API_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
_EXPECTED_API_KEY: str = os.environ.get("API_SECRET_KEY", "")


# ─── Request / Response Schemas ───────────────────────────────────────────────

class TranslateRequest(BaseModel):
    """Schema for text translation requests."""
    text: str = Field(..., min_length=1, max_length=2048, description="Source text in Devanagari")
    src_lang: str = Field(..., description="Source language code (e.g. 'mar_Deva' for Marwari)")
    tgt_lang: str = Field(..., description="Target language code (e.g. 'hin_Deva', 'eng_Latn')")


class TranslateResponse(BaseModel):
    """Schema for translation output."""
    translated_text: str
    src_lang: str
    tgt_lang: str
    model_version: str = "0.1.0"


class ASRRequest(BaseModel):
    """Schema for ASR transcription requests."""
    # 10 MB base64 ≈ 7.5 MB raw ≈ ~7 min mono 16kHz WAV — reasonable upper bound
    audio_base64: str = Field(..., max_length=10 * 1024 * 1024, description="Base64-encoded audio (16kHz WAV)")
    dialect: str = Field(..., description="Dialect hint (marwari, mewari, etc.)")


class ASRResponse(BaseModel):
    """Schema for ASR transcription output."""
    transcribed_text: str
    dialect: str
    confidence: float = 0.0


class TTSRequest(BaseModel):
    """Schema for TTS synthesis requests."""
    text: str = Field(..., min_length=1, max_length=1024)
    dialect: str = Field(..., description="Target dialect for synthesis")


class TTSResponse(BaseModel):
    """Schema for TTS synthesis output."""
    audio_base64: str
    sample_rate: int = 22050
    dialect: str


# ─── API Key Authentication ──────────────────────────────────────────────────

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """
    Validates the API key using constant-time comparison to prevent timing attacks.
    Set the expected key via the API_SECRET_KEY environment variable before starting the server:
        export API_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
    """
    if not _EXPECTED_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server API key not configured. Set API_SECRET_KEY environment variable.",
        )
    # hmac.compare_digest prevents timing-oracle attacks
    if not hmac.compare_digest(x_api_key.encode("utf-8"), _EXPECTED_API_KEY.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# ─── Translation Endpoints ───────────────────────────────────────────────────

@router.post("/translate", response_model=TranslateResponse)
async def translate_text(
    payload: TranslateRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Translate text between Rajasthani dialects and Hindi/English.
    Requires X-API-Key header for authentication.
    """
    logger.info(f"Translation request: {payload.src_lang} → {payload.tgt_lang} ({len(payload.text)} chars)")

    model = getattr(request.app.state, "mt_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="MT model not loaded")

    try:
        translations = model.translate(
            [payload.text],
            src_lang=payload.src_lang,
            tgt_lang=payload.tgt_lang,
        )
        translated = translations[0] if translations else ""
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        err_msg = str(e).lower()
        if "could not be loaded" in err_msg or "gated repo" in err_msg or "not available" in err_msg or "subsystem failure" in err_msg:
            raise HTTPException(status_code=503, detail="MT model not loaded or available")
        raise HTTPException(status_code=500, detail="Translation generation failed")

    return TranslateResponse(
        translated_text=translated,
        src_lang=payload.src_lang,
        tgt_lang=payload.tgt_lang,
    )


@router.post("/asr", response_model=ASRResponse)
async def transcribe_audio(
    payload: ASRRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Transcribe audio to Devanagari text using Whisper ASR.
    """
    logger.info(f"ASR request: dialect={payload.dialect}, audio_len={len(payload.audio_base64)}")

    model = getattr(request.app.state, "asr_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded")

    try:
        import soundfile as sf
        audio_bytes = base64.b64decode(payload.audio_base64)
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Audio must be mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        transcribed = model.transcribe_array(audio_data, sampling_rate=sr, language="hi")
    except Exception as e:
        logger.error(f"ASR failed: {e}")
        raise HTTPException(status_code=500, detail="Audio transcription failed")

    return ASRResponse(
        transcribed_text=transcribed,
        dialect=payload.dialect,
        confidence=0.0,  # Whisper does not expose per-sequence confidence natively; 0.0 = unknown
    )


@router.post("/tts", response_model=TTSResponse)
async def synthesize_speech(
    payload: TTSRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Synthesize speech from text using Indic-TTS FastPitch + HiFi-GAN.
    """
    logger.info(f"TTS request: dialect={payload.dialect}, text_len={len(payload.text)}")

    model = getattr(request.app.state, "tts_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    try:
        import soundfile as sf
        audio_array = model.synthesize(payload.text)
        
        # Convert to WAV and base64 encode
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, model.sample_rate, format="WAV")
        audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail="Speech synthesis failed")

    return TTSResponse(
        audio_base64=audio_base64,
        sample_rate=model.sample_rate,
        dialect=payload.dialect,
    )

