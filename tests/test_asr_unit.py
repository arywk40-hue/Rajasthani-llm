"""
Unit tests for WhisperASR.

All tests run without downloading any model — they test init, config
fallback, empty-input handling, and training args generation.
"""

import pytest
from src.asr.model import WhisperASR, WHISPER_MODELS


# ─── WHISPER_MODELS registry ─────────────────────────────────────────────────

def test_model_registry_contains_indicwhisper():
    assert "indicwhisper-hindi-large-v2" in WHISPER_MODELS

def test_model_registry_resolves_hf_id():
    assert WHISPER_MODELS["indicwhisper-hindi-large-v2"] == "vasista22/whisper-hindi-large-v2"


# ─── Init ────────────────────────────────────────────────────────────────────

def test_init_default_config():
    asr = WhisperASR(config_path="config/asr.yaml")
    assert asr._device in ("cpu", "cuda")
    assert asr._hf_model_id == "vasista22/whisper-hindi-large-v2"

def test_init_missing_config_does_not_crash():
    """Missing config should fall back to defaults, not raise."""
    asr = WhisperASR(config_path="config/nonexistent_asr.yaml")
    assert asr.config == {}

def test_init_custom_model_name():
    asr = WhisperASR(model_name="whisper-tiny")
    assert asr._hf_model_id == "openai/whisper-tiny"

def test_init_unknown_model_name_passthrough():
    """Unknown model names should be passed through as-is (custom HF paths)."""
    asr = WhisperASR(model_name="my-org/my-custom-whisper")
    assert asr._hf_model_id == "my-org/my-custom-whisper"

def test_model_not_loaded_before_inference():
    """Model should be None until _ensure_loaded is called."""
    asr = WhisperASR()
    assert asr._model is None
    assert asr._processor is None


# ─── Transcribe edge cases (no model load) ───────────────────────────────────

def test_transcribe_empty_list_returns_empty():
    """transcribe([]) should return [] without loading the model."""
    asr = WhisperASR()
    result = asr.transcribe([])
    assert result == []
    assert asr._model is None  # Model should NOT have been loaded


# ─── Training args ───────────────────────────────────────────────────────────

def test_get_training_args_returns_dict():
    asr = WhisperASR()
    args = asr.get_training_args()
    assert isinstance(args, dict)

def test_get_training_args_required_keys():
    asr = WhisperASR()
    args = asr.get_training_args()
    for key in ["output_dir", "num_train_epochs", "learning_rate", "predict_with_generate"]:
        assert key in args, f"Missing expected training arg: {key}"

def test_get_training_args_fp16_disabled_on_cpu():
    """FP16 should be disabled when device is CPU."""
    asr = WhisperASR(device="cpu")
    args = asr.get_training_args(fp16=True)
    assert args["fp16"] is False  # device != "cuda" → fp16 forced off
