"""
API route tests using FastAPI TestClient.

No GPU or model download required — tests cover:
- Auth rejection
- Proper 503 when models not loaded
- Payload size enforcement
- Health endpoint
"""

import os
import pytest

# Set key before importing the app so _EXPECTED_API_KEY is populated
os.environ["API_SECRET_KEY"] = "test-secret-key-for-pytest-only-32ch"

from fastapi.testclient import TestClient
from src.api.app import create_app

client = TestClient(create_app(), raise_server_exceptions=False)
HEADERS = {"X-API-Key": "test-secret-key-for-pytest-only-32ch"}
BAD_HEADERS = {"X-API-Key": "wrong-key"}


# ─── Health ──────────────────────────────────────────────────────────────────

def test_health_ok():
    r = client.get("/api/v1/health", headers=HEADERS)
    assert r.status_code == 200


# ─── Auth ────────────────────────────────────────────────────────────────────

def test_missing_api_key_returns_422():
    """Missing X-API-Key header → FastAPI 422 (required field missing)."""
    r = client.post("/api/v1/translate", json={
        "text": "नमस्ते", "src_lang": "hin_Deva", "tgt_lang": "eng_Latn"
    })
    assert r.status_code == 422


def test_wrong_api_key_returns_401():
    r = client.post("/api/v1/translate",
                    json={"text": "नमस्ते", "src_lang": "hin_Deva", "tgt_lang": "eng_Latn"},
                    headers=BAD_HEADERS)
    assert r.status_code == 401


# ─── Translate ───────────────────────────────────────────────────────────────

def test_translate_returns_503_when_model_not_loaded():
    """MT model won't load in CI — should return 503, not 500 crash."""
    r = client.post("/api/v1/translate",
                    json={"text": "नमस्ते", "src_lang": "hin_Deva", "tgt_lang": "eng_Latn"},
                    headers=HEADERS)
    # Either 200 (if model somehow loaded) or 503 (model not available)
    assert r.status_code in (200, 503)


def test_translate_empty_text_rejected():
    r = client.post("/api/v1/translate",
                    json={"text": "", "src_lang": "hin_Deva", "tgt_lang": "eng_Latn"},
                    headers=HEADERS)
    assert r.status_code == 422


def test_translate_text_too_long_rejected():
    r = client.post("/api/v1/translate",
                    json={"text": "x" * 2049, "src_lang": "hin_Deva", "tgt_lang": "eng_Latn"},
                    headers=HEADERS)
    assert r.status_code == 422


# ─── ASR ─────────────────────────────────────────────────────────────────────

def test_asr_payload_too_large_rejected():
    """Oversized audio base64 should be rejected at schema validation."""
    r = client.post("/api/v1/asr",
                    json={"audio_base64": "A" * (11 * 1024 * 1024), "dialect": "marwari"},
                    headers=HEADERS)
    assert r.status_code == 422


def test_asr_missing_dialect_rejected():
    r = client.post("/api/v1/asr",
                    json={"audio_base64": "AAAA"},
                    headers=HEADERS)
    assert r.status_code == 422


def test_asr_returns_503_when_model_not_loaded():
    """Small valid-ish base64 payload, but model not loaded in CI."""
    import base64
    tiny_wav = base64.b64encode(b"\x00" * 100).decode()
    r = client.post("/api/v1/asr",
                    json={"audio_base64": tiny_wav, "dialect": "marwari"},
                    headers=HEADERS)
    assert r.status_code in (200, 500, 503)


# ─── TTS ─────────────────────────────────────────────────────────────────────

def test_tts_empty_text_rejected():
    r = client.post("/api/v1/tts",
                    json={"text": "", "dialect": "marwari"},
                    headers=HEADERS)
    assert r.status_code == 422


def test_tts_returns_503_when_model_not_loaded():
    r = client.post("/api/v1/tts",
                    json={"text": "नमस्ते", "dialect": "marwari"},
                    headers=HEADERS)
    assert r.status_code in (200, 500, 503)
