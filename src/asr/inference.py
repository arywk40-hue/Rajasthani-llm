"""
ASR Inference Module — Whisper-based

Handles audio transcription using fine-tuned Whisper models.
"""

import torch
from loguru import logger
from src.asr.model import WhisperASR


class ASRInference:
    """Inference wrapper for Whisper ASR model."""
    
    def __init__(
        self, 
        model_path: str = None, 
        model_name: str = "indicwhisper-hindi-large-v2",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        
        if model_path:
            # Load from local checkpoint
            self.asr = WhisperASR()
            self.asr.load_checkpoint(model_path)
        else:
            # Load pretrained model
            self.asr = WhisperASR(model_name=model_name, device=device)
        
        logger.info(f"ASRInference ready on {self.device} | model={model_name}")

    @torch.no_grad()
    def transcribe(self, audio_path: str, language: str = "hi") -> str:
        """Transcribes a single audio file."""
        result = self.asr.transcribe([audio_path], language=language)
        return result[0] if result else ""

    @torch.no_grad()
    def transcribe_batch(self, audio_paths: list[str], language: str = "hi") -> list[str]:
        """Transcribes multiple audio files."""
        return self.asr.transcribe(audio_paths, language=language)
