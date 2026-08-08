"""
Unit tests for IndicTrans2MT.

All tests run in skeleton mode — no model download required.
"""

import pytest
from src.mt.model import IndicTrans2MT, _get_flores_code


# ─── FLORES code mapping ──────────────────────────────────────────────────────

def test_flores_hindi():
    assert _get_flores_code("hindi") == "hin_Deva"

def test_flores_english():
    assert _get_flores_code("english") == "eng_Latn"

def test_flores_dialect_proxies():
    """All 6 Rajasthani dialects should proxy to hin_Deva."""
    for dialect in ["marwari", "mewari", "dhundhari", "hadoti", "mewati", "bagri"]:
        assert _get_flores_code(dialect) == "hin_Deva", f"{dialect} should map to hin_Deva"

def test_flores_passthrough():
    """Already-formatted FLORES codes should pass through unchanged."""
    assert _get_flores_code("hin_Deva") == "hin_Deva"
    assert _get_flores_code("eng_Latn") == "eng_Latn"

def test_flores_unknown_defaults_to_hindi():
    """Unknown language should fall back to hin_Deva."""
    assert _get_flores_code("unknown_lang_xyz") == "hin_Deva"


# ─── IndicTrans2MT init ──────────────────────────────────────────────────────

def test_init_missing_config_does_not_crash():
    """Missing config file should log a warning and use defaults, not crash."""
    model = IndicTrans2MT(config_path="config/nonexistent_mt.yaml")
    assert model.config == {}

def test_init_with_valid_config():
    model = IndicTrans2MT(config_path="config/mt.yaml")
    assert isinstance(model.config, dict)


# ─── Skeleton mode ───────────────────────────────────────────────────────────

def test_skeleton_init():
    model = IndicTrans2MT(config_path="config/mt.yaml")
    model._init_skeleton()
    assert model._skeleton_model is not None

def test_skeleton_translate_returns_placeholders():
    """Skeleton translate should return placeholder strings, not crash."""
    model = IndicTrans2MT(config_path="config/mt.yaml")
    model._init_skeleton()
    results = model.translate(["नमस्ते"], src_lang="hindi", tgt_lang="english")
    assert len(results) == 1
    assert isinstance(results[0], str)
    assert len(results[0]) > 0

def test_skeleton_translate_empty_list():
    model = IndicTrans2MT(config_path="config/mt.yaml")
    model._init_skeleton()
    results = model.translate([], src_lang="hindi", tgt_lang="english")
    assert results == []

def test_skeleton_is_loaded_false():
    model = IndicTrans2MT(config_path="config/mt.yaml")
    model._init_skeleton()
    assert model.is_loaded is False


# ─── Trainable parameters ────────────────────────────────────────────────────

def test_get_trainable_parameters_skeleton():
    model = IndicTrans2MT(config_path="config/mt.yaml")
    model._init_skeleton()
    n = model.get_trainable_parameters()
    assert isinstance(n, int)
    assert n > 0
