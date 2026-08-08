"""
MT Pipeline Module

Provides implementations for the IndicTrans2 Machine Translation architecture,
including models, training with experience replay, and back-translation data augmentation.
"""

from src.mt.model import IndicTrans2MT
from src.mt.trainer import MTTrainer
from src.mt.backtranslation import BackTranslationGenerator

__all__ = ["IndicTrans2MT", "MTTrainer", "BackTranslationGenerator"]
