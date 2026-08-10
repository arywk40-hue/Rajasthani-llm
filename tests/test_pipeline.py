"""
Integration tests for the S2ST Pipeline orchestration.
"""

import pytest
from pathlib import Path
from src.pipeline.pipeline import SpeechToSpeechPipeline
from src.mt.pipeline import DialectNormalizer, TranslationPipeline


def test_dialect_normalizer():
    """Verify that DialectNormalizer successfully maps dialect terms to Hindi equivalents."""
    normalizer = DialectNormalizer()
    
    # Marwari mapping test
    marwari_text = "टाबर कठै जा रयो है"
    mapped_marwari = normalizer.normalize_to_hindi(marwari_text, "marwari")
    assert "बच्चे" in mapped_marwari
    assert "कहाँ" in mapped_marwari
    
    # Dhundhari mapping test
    dhundhari_text = "छोरा कठै जा रयो छै"
    mapped_dhundhari = normalizer.normalize_to_hindi(dhundhari_text, "dhundhari")
    assert "लड़का" in mapped_dhundhari
    assert "है" in mapped_dhundhari


def test_pipeline_instantiation():
    """Verify we can construct the pipeline without exceptions."""
    # We can instantiate with empty/default args or test mock loading
    # To run in CI quickly, we can test just DialectNormalizer and translation abstractions
    pipeline = TranslationPipeline(model_name="indic-indic-dist-320M")
    assert pipeline.normalizer is not None
    assert pipeline.mt_model is not None
