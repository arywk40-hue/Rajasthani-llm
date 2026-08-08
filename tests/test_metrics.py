import pytest
from src.evaluation.metrics import calculate_wer, calculate_cer, calculate_chrf, calculate_bleu

def test_calculate_wer():
    references = ["hello world", "this is a test"]
    hypotheses = ["hello world", "this is test"]
    wer = calculate_wer(references, hypotheses)
    assert isinstance(wer, float)
    assert 0.0 <= wer <= 1.0

def test_calculate_cer():
    references = ["hello world", "this is a test"]
    hypotheses = ["hello world", "this is test"]
    cer = calculate_cer(references, hypotheses)
    assert isinstance(cer, float)
    assert 0.0 <= cer <= 1.0

def test_calculate_chrf():
    references = ["hello world", "this is a test"]
    hypotheses = ["hello world", "this is test"]
    chrf = calculate_chrf(references, hypotheses)
    assert isinstance(chrf, float)

def test_calculate_bleu():
    references = ["hello world", "this is a test"]
    hypotheses = ["hello world", "this is test"]
    bleu = calculate_bleu(references, hypotheses)
    assert isinstance(bleu, float)
