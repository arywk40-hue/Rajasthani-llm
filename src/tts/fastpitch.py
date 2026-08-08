"""
Indic-TTS Integration — FastPitch + HiFi-GAN for Rajasthani

AI4Bharat's Indic-TTS already has a Rajasthani model!
We load it directly and fine-tune per dialect (1-2 hrs of speaker recordings).

Architecture: FastPitch (acoustic model) + HiFi-GAN V1 (vocoder)
- FastPitch: non-autoregressive transformer with pitch + duration control
- HiFi-GAN: mel-spectrogram → waveform in real-time

Strategy from hackathon plan:
- Fork AI4Bharat/Indic-TTS Rajasthani model
- Fine-tune FastPitch on per-dialect speaker recordings
- Even 1-2 hours per dialect is enough for a demo

Reference: detailed-report.md Phase 2 (TTS subsystem)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml
from loguru import logger


# Model registry for Indic TTS
INDIC_TTS_MODELS = {
    # NeMo-based models (if NeMo is installed)
    "rajasthani_fastpitch": "ai4bharat/indic-tts-rajasthani-fastpitch",
    "hindi_fastpitch": "ai4bharat/indic-tts-hindi-fastpitch",
    # HiFi-GAN vocoders
    "hifigan_v1": "ai4bharat/indic-tts-hifigan",
}


class IndicTTS:
    """
    Text-to-Speech engine for Rajasthani dialects.

    Wraps AI4Bharat's Indic-TTS models (FastPitch + HiFi-GAN).
    Supports both NeMo and direct PyTorch loading.

    Usage:
        tts = IndicTTS()

        # Synthesize speech
        audio = tts.synthesize("राजस्थानी भाषा बहुत सुंदर है", dialect="marwari")

        # Save to file
        tts.synthesize_to_file(
            "होनो कहते हैं",
            output_path="output.wav",
            dialect="marwari",
        )
    """

    def __init__(
        self,
        config_path: str = "config/tts.yaml",
        device: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.config = cfg.get("tts", {})
        else:
            self.config = {}

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded models
        self._fastpitch = None
        self._hifigan = None
        self._nemo_available = False
        self._loaded = False

        self.sample_rate = self.config.get("vocoder", {}).get("sample_rate", 22050)

        logger.info(f"IndicTTS initialized | device={self._device}")

    def _load_models(self):
        """Load TTS models (FastPitch + HiFi-GAN)."""
        if self._loaded:
            return

        # Try NeMo first (has pretrained Indic models)
        try:
            self._load_nemo_models()
            self._nemo_available = True
            self._loaded = True
            return
        except (ImportError, Exception) as e:
            logger.info(f"NeMo not available ({e}), trying HuggingFace...")

        # Fallback: Try HuggingFace TTS
        try:
            self._load_hf_models()
            self._loaded = True
            return
        except (ImportError, Exception) as e:
            logger.info(f"HuggingFace TTS not available ({e}), using skeleton...")

        # Final fallback: skeleton
        self._loaded = True
        logger.warning(
            "No TTS backend available. Install NeMo or HuggingFace TTS. "
            "Outputs will be silent audio."
        )

    def _load_nemo_models(self):
        """Load NeMo FastPitch + HiFi-GAN models."""
        import nemo.collections.tts as nemo_tts

        logger.info("Loading NeMo FastPitch model...")
        # Try Rajasthani first, fall back to Hindi
        try:
            self._fastpitch = nemo_tts.models.FastPitchModel.from_pretrained(
                "ai4bharat/indic-tts-rajasthani-fastpitch"
            )
        except Exception:
            logger.info("Rajasthani FastPitch not found, using Hindi base...")
            self._fastpitch = nemo_tts.models.FastPitchModel.from_pretrained(
                "tts_hi_fastpitch"
            )

        self._fastpitch.to(self._device)
        self._fastpitch.eval()

        logger.info("Loading NeMo HiFi-GAN vocoder...")
        self._hifigan = nemo_tts.models.HifiGanModel.from_pretrained(
            "tts_en_hifigan"  # HiFi-GAN is language-agnostic for mel→audio
        )
        self._hifigan.to(self._device)
        self._hifigan.eval()

        logger.success("NeMo TTS models loaded (FastPitch + HiFi-GAN)")

    def _load_hf_models(self):
        """Load TTS via HuggingFace transformers (SpeechT5 or similar)."""
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

        logger.info("Loading HuggingFace SpeechT5 as TTS fallback...")

        self._hf_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self._fastpitch = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self._hifigan = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        self._fastpitch.to(self._device)
        self._hifigan.to(self._device)

        # Load speaker embeddings
        try:
            from datasets import load_dataset
            embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors", split="validation"
            )
            self._speaker_embedding = torch.tensor(
                embeddings_dataset[7306]["xvector"]
            ).unsqueeze(0).to(self._device)
        except Exception:
            self._speaker_embedding = torch.randn(1, 512).to(self._device)

        self._nemo_available = False
        self.sample_rate = 16000  # SpeechT5 uses 16kHz
        logger.success("HuggingFace SpeechT5 TTS loaded")

    # ─── Synthesis ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        dialect: Optional[str] = None,
        speed: float = 1.0,
    ):
        """
        Synthesize speech from text.

        Args:
            text: Input text in Devanagari
            dialect: Optional dialect tag (for future per-dialect models)
            speed: Speech rate multiplier (1.0 = normal)

        Returns:
            numpy array of audio samples at self.sample_rate Hz
        """
        self._load_models()

        if self._fastpitch is None:
            # Skeleton: return silent audio
            import numpy as np
            duration = max(len(text) * 0.1, 1.0)  # ~100ms per char
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

        if self._nemo_available:
            return self._synthesize_nemo(text, speed)
        else:
            return self._synthesize_hf(text)

    def _synthesize_nemo(self, text: str, speed: float = 1.0):
        """Synthesize using NeMo FastPitch + HiFi-GAN."""
        import numpy as np

        # Generate mel spectrogram
        parsed = self._fastpitch.parse(text)
        spectrogram = self._fastpitch.generate_spectrogram(tokens=parsed)

        # Apply speed control
        if speed != 1.0:
            import torch.nn.functional as F
            target_len = int(spectrogram.shape[-1] / speed)
            spectrogram = F.interpolate(
                spectrogram, size=target_len, mode="linear", align_corners=False
            )

        # Generate waveform
        audio = self._hifigan.convert_spectrogram_to_audio(spec=spectrogram)
        return audio.squeeze().cpu().numpy()

    def _synthesize_hf(self, text: str):
        """Synthesize using HuggingFace SpeechT5."""
        import numpy as np

        inputs = self._hf_processor(text=text, return_tensors="pt").to(self._device)
        speech = self._fastpitch.generate_speech(
            inputs["input_ids"],
            self._speaker_embedding,
            vocoder=self._hifigan,
        )
        return speech.cpu().numpy()

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        dialect: Optional[str] = None,
        speed: float = 1.0,
    ) -> Path:
        """Synthesize speech and save to WAV file."""
        import soundfile as sf

        audio = self.synthesize(text, dialect=dialect, speed=speed)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.sample_rate)
        logger.info(f"Synthesized speech saved to {output_path}")
        return output_path

    def synthesize_batch(
        self,
        texts: list[str],
        output_dir: str | Path,
        dialect: Optional[str] = None,
    ) -> list[Path]:
        """Synthesize multiple texts to WAV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []
        for i, text in enumerate(texts):
            out_path = output_dir / f"tts_{i:04d}.wav"
            self.synthesize_to_file(text, out_path, dialect=dialect)
            output_paths.append(out_path)

        logger.info(f"Synthesized {len(output_paths)} audio files to {output_dir}")
        return output_paths
