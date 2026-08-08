"""
IndicTrans2 Translation Inference Wrapper

Provides a clean inference API for MT, mirroring WhisperASR pattern.
Loads the HF model, handles IndicProcessor preprocessing, and exposes
a simple translate() method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml
from loguru import logger

from src.mt.model import IndicTrans2MT


class IndicTrans2Translator:
    """
    Production inference wrapper for IndicTrans2.

    Usage:
        translator = IndicTrans2Translator()
        translations = translator.translate(
            ["नमस्ते, आप कैसे हैं?"],
            src_lang="hindi",
            tgt_lang="english"
        )

        # For dialects, use FLORES codes or dialect names
        translations = translator.translate(
            ["होनो कहते हैं"],
            src_lang="marwari",
            tgt_lang="hindi"
        )
    """

    def __init__(
        self,
        config_path: str = "config/mt.yaml",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.config = cfg.get("mt", {})
        else:
            self.config = {}

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = model_name or self.config.get("hf_model_name", "indic-indic-1B")

        self._model = IndicTrans2MT(
            config_path=str(self.config_path),
            model_name=self._model_name,
            device=self._device,
        )

        logger.info(f"IndicTrans2Translator initialized | model={self._model_name} device={self._device}")

    def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        num_beams: int = 5,
        max_length: int = 256,
    ) -> list[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of source texts
            src_lang: Source language (e.g., "hindi", "marwari", "eng_Latn")
            tgt_lang: Target language (e.g., "english", "hindi", "hin_Deva")
            num_beams: Beam search width
            max_length: Max generation length

        Returns:
            List of translated texts
        """
        return self._model.translate(
            texts=texts,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            num_beams=num_beams,
            max_length=max_length,
        )

    def translate_single(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        """Convenience method for single-string translation."""
        return self.translate([text], src_lang, tgt_lang)[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IndicTrans2 translation inference")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    parser.add_argument("--src", type=str, default="hindi", help="Source language")
    parser.add_argument("--tgt", type=str, default="english", help="Target language")
    parser.add_argument("--config", type=str, default="config/mt.yaml")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    translator = IndicTrans2Translator(config_path=args.config, model_name=args.model)
    result = translator.translate_single(args.text, args.src, args.tgt)
    print(f"Translation: {result}")