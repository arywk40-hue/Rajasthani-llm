"""
ASR Inference Module — Whisper-based

Handles audio transcription using fine-tuned Whisper models with
proper timing, sample-rate validation, and Devanagari normalization.
"""

import time
import torch
from loguru import logger
from src.asr.model import WhisperASR
from src.preprocessing.normalizer import DevanagariNormalizer


class ASRInference:
    """Inference wrapper for Whisper ASR model."""
    
    def __init__(
        self, 
        model_path: str = None, 
        model_name: str = "indicwhisper-hindi-large-v2",
        device: str = "auto",
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path:
            # Load from local checkpoint
            self.asr = WhisperASR(device=self.device)
            self.asr.load_checkpoint(model_path)
        else:
            # Load pretrained model
            self.asr = WhisperASR(model_name=model_name, device=self.device)
        
        self.normalizer = DevanagariNormalizer()
        logger.info(f"ASRInference ready on {self.device} | model={model_name}")

    def transcribe(self, audio_path: str, language: str = "hi") -> dict:
        """
        Transcribes a single audio file and returns a structured dict with timing and normalization.
        """
        import soundfile as sf
        
        start_time = time.perf_counter()
        
        try:
            # Validate sample rate and file readability
            info = sf.info(audio_path)
            if info.samplerate != 16000:
                logger.warning(f"Audio sample rate is {info.samplerate}Hz (expected 16000Hz). Will resample.")
            
            # Run inference
            raw_transcript_list = self.asr.transcribe([audio_path], language=language)
            raw_transcript = raw_transcript_list[0] if raw_transcript_list else ""
        except Exception as e:
            logger.error(f"ASR Transcription failed for {audio_path}: {e}")
            raise RuntimeError(f"ASR Subsystem Failure: {e}")
        
        # Apply Devanagari normalisation
        normalized_transcript = self.normalizer.normalize(raw_transcript)
        
        elapsed = time.perf_counter() - start_time
        
        return {
            "audio": str(audio_path),
            "transcript": raw_transcript,
            "normalized_text": normalized_transcript,
            "latency_ms": round(elapsed * 1000, 2)
        }

    def transcribe_batch(self, audio_paths: list[str], language: str = "hi") -> list[dict]:
        """
        Transcribes multiple audio files in a batch.
        """
        start_time = time.perf_counter()
        
        try:
            raw_transcripts = self.asr.transcribe(audio_paths, language=language)
        except Exception as e:
            logger.error(f"ASR Batch Transcription failed: {e}")
            raise RuntimeError(f"ASR Subsystem Failure: {e}")
            
        elapsed = time.perf_counter() - start_time
        per_sample_latency = (elapsed * 1000) / max(len(audio_paths), 1)
        
        results = []
        for path, raw in zip(audio_paths, raw_transcripts):
            results.append({
                "audio": str(path),
                "transcript": raw,
                "normalized_text": self.normalizer.normalize(raw),
                "latency_ms": round(per_sample_latency, 2)
            })
        return results

