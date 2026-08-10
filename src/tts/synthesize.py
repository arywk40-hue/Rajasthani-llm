"""
Text-to-Speech Subsystem — local VITS-based

Synthesizes speech from Devanagari text using Meta's Massively Multilingual
Speech (MMS) VITS model for Hindi. Since a true Rajasthani local TTS model
does not exist, we use the Hindi VITS baseline (labeled as BASELINE).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger


class IndicTTSSynthesizer:
    """
    Local Text-to-Speech synthesis using Meta VITS.
    Uses 'facebook/mms-tts-hin' as a local, API-key-free baseline model.
    """

    def __init__(
        self,
        config_path: str = "config/tts.yaml",
        model_name: str = "facebook/mms-tts-hin",
        device: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        if device == "auto" or device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._model_name = model_name
        
        self._tokenizer = None
        self._model = None
        self._sample_rate = 16000  # MMS VITS operates at 16kHz
        
        logger.info(f"IndicTTSSynthesizer initialized | model={model_name} device={self._device}")

    def _ensure_loaded(self):
        """Lazy-load the VITS model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import VitsModel, AutoTokenizer

            logger.info(f"Loading local VITS model: {self._model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = VitsModel.from_pretrained(self._model_name).to(self._device)
            logger.success(f"VITS Model loaded successfully: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to load VITS model {self._model_name}: {e}")
            raise RuntimeError(f"TTS Subsystem Failure: {e}")

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """
        Synthesize text into speech WAV waveform.

        Returns:
            1D array of audio samples at 16000 Hz
        """
        self._ensure_loaded()
        
        if not text.strip():
            import numpy as np
            return np.zeros(0, dtype=np.float32)

        try:
            # Process input text
            inputs = self._tokenizer(text=text, return_tensors="pt").to(self._device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # VITS outputs waveform shape: [1, seq_len]
                waveform = outputs.waveform[0].cpu()
                
            if return_numpy:
                return waveform.numpy()
            return waveform
        except Exception as e:
            logger.error(f"VITS Speech synthesis failed: {e}")
            raise RuntimeError(f"TTS Subsystem Failure: {e}")

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
    ) -> Path:
        """Synthesize speech and save to WAV file."""
        import soundfile as sf

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        audio = self.synthesize(text)
        sf.write(str(output_path), audio, self.sample_rate)
        logger.info(f"Synthesized speech saved to {output_path}")
        return output_path

    def synthesize_batch(
        self,
        texts: list[str],
    ) -> list[np.ndarray]:
        """Synthesize multiple texts."""
        return [self.synthesize(t) for t in texts]

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_id(self) -> str:
        return self._model_name