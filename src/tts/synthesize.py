"""
Indic-TTS Synthesis Wrapper

Loads AI4Bharat Indic-TTS models for Rajasthani dialects.
Uses FastPitch (acoustic) + HiFi-GAN (vocoder) pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml
from loguru import logger


class IndicTTSSynthesizer:
    """
    Production inference wrapper for Indic-TTS.

    Supports the AI4Bharat IndicTTS Rajasthani model which covers
    multiple dialects under one "Rajasthani" language code.

    Usage:
        tts = IndicTTSSynthesizer()
        audio_array = tts.synthesize("नमस्ते, राजस्थानी में आपका स्वागत है")
        
        # Save to file
        import soundfile as sf
        sf.write("output.wav", audio_array, 22050)
    """

    # AI4Bharat IndicTTS model registry
    MODEL_REGISTRY = {
        "rajasthani": "ai4bharat/indic-tts-rajasthani-fastpitch-hifigan",
        "hindi": "ai4bharat/indic-tts-hindi-fastpitch-hifigan",
    }

    def __init__(
        self,
        config_path: str = "config/tts.yaml",
        model_name: str = "rajasthani",
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
        self._model_name = model_name
        self._hf_model_id = self.MODEL_REGISTRY.get(model_name, model_name)

        # Lazy-loaded components
        self._fastpitch = None
        self._hifigan = None
        self._processor = None

        logger.info(f"IndicTTSSynthesizer initialized | model={self._hf_model_id} device={self._device}")

    def _ensure_loaded(self):
        """Lazy-load the TTS models from HuggingFace."""
        if self._fastpitch is not None and self._hifigan is not None:
            return

        try:
            from transformers import AutoModel, AutoProcessor

            logger.info(f"Loading Indic-TTS model: {self._hf_model_id}")

            self._processor = AutoProcessor.from_pretrained(self._hf_model_id)
            self._fastpitch = AutoModel.from_pretrained(
                self._hf_model_id, subfolder="fastpitch"
            ).to(self._device)
            self._hifigan = AutoModel.from_pretrained(
                self._hf_model_id, subfolder="hifigan"
            ).to(self._device)

            logger.success(f"Indic-TTS loaded: {self._hf_model_id}")

        except Exception as e:
            logger.error(f"Failed to load Indic-TTS model: {e}")
            logger.warning("Using fallback synthesis (silence) — install transformers and check model ID")
            self._fastpitch = None
            self._hifigan = None

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        return_numpy: bool = True,
    ):
        """
        Synthesize speech from text.

        Args:
            text: Input text in Devanagari
            speaker_id: Speaker index (if multi-speaker model)
            return_numpy: Return numpy array instead of torch tensor

        Returns:
            Audio waveform (numpy array or torch tensor) at 22050 Hz
        """
        self._ensure_loaded()

        if self._fastpitch is None or self._hifigan is None:
            raise RuntimeError(
                "TTS models are not loaded. Check server logs for the load error. "
                "Ensure the model ID is valid or configure Bhashini API key."
            )

        # Process text through processor
        inputs = self._processor(
            text=text,
            speaker_id=speaker_id,
            return_tensors="pt",
        ).to(self._device)

        # FastPitch: text → mel spectrogram
        mel_output = self._fastpitch.generate(**inputs)

        # HiFi-GAN: mel → waveform
        waveform = self._hifigan(mel_output).squeeze()

        if return_numpy:
            return waveform.cpu().numpy()
        return waveform

    def synthesize_batch(
        self,
        texts: list[str],
        speaker_id: int = 0,
    ) -> list:
        """Synthesize multiple texts."""
        return [self.synthesize(t, speaker_id) for t in texts]

    @property
    def sample_rate(self) -> int:
        return 22050


if __name__ == "__main__":
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Indic-TTS synthesis")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--config", type=str, default="config/tts.yaml")
    parser.add_argument("--model", type=str, default="rajasthani")
    args = parser.parse_args()

    tts = IndicTTSSynthesizer(config_path=args.config, model_name=args.model)
    audio = tts.synthesize(args.text)
    sf.write(args.output, audio, tts.sample_rate)
    print(f"Saved to {args.output}")