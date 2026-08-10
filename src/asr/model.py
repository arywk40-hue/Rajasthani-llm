"""
Whisper-based ASR for Rajasthani Dialects

Replaces the FastConformer skeleton with a real Whisper fine-tuning pipeline.
Uses vasista22/whisper-hindi-large-v2 (IndicWhisper) as the base model —
pretrained on 10,700+ hours of Hindi via the Vistaar benchmark.

Fine-tuning strategy (from hackathon 3-track plan):
1. Zero-shot baseline: Run IndicWhisper on VAANI Rajasthan data → measure WER
2. Joint fine-tune: Pool all 6 dialects into "Rajasthani dialect" fine-tune
3. Per-dialect adapt: Fine-tune per dialect with dialect-specific data

Reference: IndicWhisper achieves lowest WER on 39/59 Indic ASR benchmarks.
Even 10-50 hrs of dialect data gives 20-40% relative WER reduction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml
from loguru import logger


# ─── Pretrained Model Registry ───────────────────────────────────────────────

WHISPER_MODELS = {
    # IndicWhisper — Hindi fine-tuned, best for Rajasthani dialects
    "indicwhisper-hindi-large-v2": "vasista22/whisper-hindi-large-v2",
    "indicwhisper-hindi-medium": "vasista22/whisper-hindi-medium",
    "indicwhisper-hindi-small": "vasista22/whisper-hindi-small",
    # Base OpenAI Whisper
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-large-v2": "openai/whisper-large-v2",
    "whisper-medium": "openai/whisper-medium",
    "whisper-small": "openai/whisper-small",
    "whisper-base": "openai/whisper-base",
    "whisper-tiny": "openai/whisper-tiny",
    # Sarvam AI — NOTE: Saaras is API-only, not a public HF checkpoint
    # "saaras-v1": "sarvamai/saaras-v1",  # unavailable on HF
}


class WhisperASR:
    """
    Whisper-based ASR for Rajasthani dialect transcription.

    Supports:
    - Zero-shot inference with pretrained IndicWhisper
    - Fine-tuning on dialect-specific audio data
    - Batch transcription with language forcing (Hindi → catches dialects)
    - CTC/attention decoding

    Usage:
        # Zero-shot inference
        asr = WhisperASR()
        transcripts = asr.transcribe(["audio1.wav", "audio2.wav"])

        # Fine-tuning
        asr = WhisperASR(model_name="indicwhisper-hindi-large-v2")
        asr.prepare_for_finetuning()
        # ... use asr.model and asr.processor with HF Trainer
    """

    def __init__(
        self,
        config_path: str = "config/asr.yaml",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.config = cfg.get("asr", {})
        else:
            self.config = {}

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = model_name or self.config.get(
            "hf_model_name", "indicwhisper-hindi-large-v2"
        )

        # Resolve model name
        self._hf_model_id = WHISPER_MODELS.get(self._model_name, self._model_name)

        # Lazy-loaded components
        self._model = None
        self._processor = None
        self._feature_extractor = None

        logger.info(
            f"WhisperASR initialized | model={self._hf_model_id} device={self._device}"
        )

    # ─── Model Loading ────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Lazy-load the model and processor."""
        if self._model is not None:
            return

        try:
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
                WhisperFeatureExtractor,
            )

            logger.info(f"Loading Whisper model: {self._hf_model_id}")

            self._processor = WhisperProcessor.from_pretrained(
                self._hf_model_id, trust_remote_code=True
            )
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self._hf_model_id
            )
            self._model = WhisperForConditionalGeneration.from_pretrained(
                self._hf_model_id,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            self._model.to(self._device)

            logger.success(f"Whisper loaded: {self._hf_model_id}")

        except ImportError:
            logger.error(
                "transformers not installed. Install with: "
                "pip install transformers[torch]"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    @property
    def model_id(self) -> str:
        """The resolved HuggingFace model id, for reporting in benchmark output."""
        return self._hf_model_id

    @property
    def model(self):
        """Access the underlying Whisper model (for HF Trainer integration)."""
        self._ensure_loaded()
        return self._model

    @property
    def processor(self):
        """Access the Whisper processor (for HF Trainer integration)."""
        self._ensure_loaded()
        return self._processor

    @property
    def feature_extractor(self):
        """Access the feature extractor."""
        self._ensure_loaded()
        return self._feature_extractor

    # ─── Inference ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def transcribe(
        self,
        audio_paths: list[str | Path],
        language: str = "hi",
        task: str = "transcribe",
        batch_size: int = 8,
    ) -> list[str]:
        """
        Transcribe audio files to text.

        Args:
            audio_paths: Paths to WAV/FLAC audio files
            language: Language code for Whisper (use "hi" for all Rajasthani dialects)
            task: "transcribe" or "translate" (translate → English)
            batch_size: Number of files to process at once

        Returns:
            List of transcribed text strings
        """
        if not audio_paths:
            return []
        self._ensure_loaded()

        try:
            import soundfile as sf
            import numpy as np
        except ImportError:
            logger.error("soundfile and numpy are required. Install with: pip install soundfile numpy")
            raise

        transcripts = []
        forced_decoder_ids = self._processor.get_decoder_prompt_ids(
            language=language, task=task
        )

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i : i + batch_size]
            batch_audio = []

            for path in batch_paths:
                try:
                    audio, sr = sf.read(str(path))
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    if sr != 16000:
                        # numpy resampling
                        num_samples = int(len(audio) * 16000 / sr)
                        audio = np.interp(
                            np.linspace(0, len(audio), num_samples, endpoint=False),
                            np.arange(len(audio)),
                            audio
                        )
                    batch_audio.append(audio)
                except Exception as e:
                    logger.warning(f"Could not load {path}: {e}")
                    batch_audio.append(None)

            # Filter out failed loads
            valid_audio = [a for a in batch_audio if a is not None]
            if not valid_audio:
                transcripts.extend(["" for _ in batch_paths])
                continue

            # Process through Whisper
            input_features = self._processor(
                valid_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).input_features.to(self._device)

            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
            )

            batch_transcripts = self._processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            # Re-insert empty strings for failed loads
            result_idx = 0
            for audio in batch_audio:
                if audio is not None:
                    transcripts.append(batch_transcripts[result_idx].strip())
                    result_idx += 1
                else:
                    transcripts.append("")

        logger.info(f"Transcribed {len(transcripts)} audio files")
        return transcripts

    @torch.inference_mode()
    def transcribe_array(
        self,
        audio_array,
        sampling_rate: int = 16000,
        language: str = "hi",
    ) -> str:
        """Transcribe a single audio numpy array."""
        self._ensure_loaded()

        if sampling_rate != 16000:
            import numpy as np
            # numpy resampling
            num_samples = int(len(audio_array) * 16000 / sampling_rate)
            audio_array = np.interp(
                np.linspace(0, len(audio_array), num_samples, endpoint=False),
                np.arange(len(audio_array)),
                audio_array
            )

        input_features = self._processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(self._device)

        forced_decoder_ids = self._processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

        predicted_ids = self._model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=448,
        )

        return self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

    # ─── Fine-tuning Setup ────────────────────────────────────────────────────

    def prepare_for_finetuning(
        self,
        freeze_encoder: bool = True,
        gradient_checkpointing: bool = True,
    ):
        """
        Prepare the model for fine-tuning on dialect data.

        Strategy from hackathon plan:
        - Freeze encoder (pretrained acoustic features are good enough)
        - Only train decoder (adapts to dialect vocabulary/phonetics)
        - Enable gradient checkpointing to reduce memory usage

        Args:
            freeze_encoder: Whether to freeze the encoder (recommended)
            gradient_checkpointing: Reduce memory at cost of speed
        """
        self._ensure_loaded()

        if gradient_checkpointing:
            self._model.config.use_cache = False
            self._model.gradient_checkpointing_enable()

        if freeze_encoder:
            for param in self._model.model.encoder.parameters():
                param.requires_grad = False
            trainable = sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self._model.parameters())
            logger.info(
                f"Encoder frozen. Trainable: {trainable:,} / {total:,} "
                f"({trainable / total * 100:.1f}%)"
            )

        # Disable forced decoder IDs for fine-tuning
        self._model.config.forced_decoder_ids = None
        self._model.config.suppress_tokens = []

        logger.success("Model prepared for fine-tuning")

    def get_training_args(
        self,
        output_dir: str = "checkpoints/asr_whisper",
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 1e-5,
        warmup_steps: int = 500,
        fp16: bool = True,
    ) -> dict:
        """
        Get recommended Seq2SeqTrainingArguments for dialect fine-tuning.
        Returns a dict that can be passed to Seq2SeqTrainingArguments(**args).
        """
        return {
            "output_dir": output_dir,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_train_batch_size * 2,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "fp16": fp16 and self._device == "cuda",
            "gradient_accumulation_steps": 4,
            "eval_strategy": "steps",
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "logging_steps": 50,
            "load_best_model_at_end": True,
            "metric_for_best_model": "cer",
            "greater_is_better": False,
            "predict_with_generate": True,
            "generation_max_length": 448,
            "report_to": "none",
            "push_to_hub": False,
            "dataloader_num_workers": 2,
            "remove_unused_columns": False,
        }

    # ─── Checkpoint Management ────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save fine-tuned model and processor."""
        self._ensure_loaded()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._processor.save_pretrained(path)
        logger.info(f"Whisper checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a fine-tuned checkpoint."""
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        path = Path(path)
        self._processor = WhisperProcessor.from_pretrained(str(path))
        self._model = WhisperForConditionalGeneration.from_pretrained(str(path))
        self._model.to(self._device)
        logger.info(f"Loaded checkpoint from {path}")

    # ─── Zero-shot Benchmark ─────────────────────────────────────────────────

    def benchmark_zero_shot(
        self,
        audio_paths: list[str | Path],
        references: list[str],
        language: str = "hi",
    ) -> dict:
        """
        Run zero-shot benchmark on dialect audio data.

        This is Track 1 of the hackathon strategy:
        Run IndicWhisper on VAANI Rajasthan data → establish WER/CER baseline.

        Returns:
            Dict with {mean_cer, mean_wer, per_sample: [...]}
        """
        from ..evaluation.metrics import compute_cer, compute_wer

        transcripts = self.transcribe(audio_paths, language=language)

        results = []
        total_cer, total_wer = 0.0, 0.0
        for hyp, ref in zip(transcripts, references):
            cer = compute_cer(hyp, ref)
            wer = compute_wer(hyp, ref)
            total_cer += cer
            total_wer += wer
            results.append({"hypothesis": hyp, "reference": ref, "cer": cer, "wer": wer})

        n = max(len(results), 1)
        report = {
            "model": self._hf_model_id,
            "samples": len(results),
            "mean_cer": total_cer / n,
            "mean_wer": total_wer / n,
            "per_sample": results,
        }

        logger.info(
            f"Zero-shot benchmark: CER={report['mean_cer']:.4f} "
            f"WER={report['mean_wer']:.4f} ({len(results)} samples)"
        )
        return report
