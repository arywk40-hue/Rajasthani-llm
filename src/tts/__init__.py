"""
TTS Pipeline Module

Provides implementations for the TTS architecture:
1. IndicTTS (AI4Bharat Indic-TTS: FastPitch + HiFi-GAN for Rajasthani)
2. IndicTTSSynthesizer (HuggingFace-based inference wrapper)
"""

from src.tts.fastpitch import IndicTTS
from src.tts.synthesize import IndicTTSSynthesizer
from src.tts.trainer import TTSTrainer

__all__ = ["IndicTTS", "IndicTTSSynthesizer", "TTSTrainer"]
