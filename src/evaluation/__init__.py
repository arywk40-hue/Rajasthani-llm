"""
Evaluation package for Rajasthani Dialect AI.

Implements roadmap §6 metrics:
- ASR: Character Error Rate (CER) — real Levenshtein-based, not word-level
- MT : chrF++ and COMET (BLEU is unreliable for morphologically rich low-resource languages)
- TTS: Mean Opinion Score (MOS) tracking + Phonetically Balanced intelligibility
"""

from src.evaluation.metrics import Evaluator

__all__ = ["Evaluator"]