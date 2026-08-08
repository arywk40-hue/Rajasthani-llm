"""
TTS Pipeline Module

Provides implementations for the two-stage TTS architecture:
1. FastPitch (Acoustic model for mel-spectrogram generation)
2. HiFi-GAN V1 (Vocoder for waveform generation)
"""

from src.tts.fastpitch import FastPitchAcoustic
from src.tts.hifigan import HiFiGANVocoder
from src.tts.trainer import TTSTrainer

__all__ = ["FastPitchAcoustic", "HiFiGANVocoder", "TTSTrainer"]
