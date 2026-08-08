import pytest
from src.evaluation.metrics import compute_wer, compute_cer, compute_chrf

def test_compute_wer():
    references = ["hello world", "this is a test"]
    hypotheses = ["hello world", "this is test"]
    wer = compute_wer(hypotheses[0], references[0])
    assert isinstance(wer, float)
    assert 0.0 <= wer <= 1.0

def test_compute_cer():
    references = ["hello world", "this is a test"]
    hypotheses = ["hello world", "this is test"]
    cer = compute_cer(hypotheses[0], references[0])
    assert isinstance(cer, float)
    assert 0.0 <= cer <= 1.0

def test_compute_chrf():
    references = ["hello world", "this is a test"]
    hypotheses = ["hello world", "this is test"]
    chrf = compute_chrf(hypotheses[0], references[0])
    assert isinstance(chrf, float)
