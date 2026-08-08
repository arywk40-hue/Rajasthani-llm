"""
ASR Fine-tuning Trainer

Handles the Whisper fine-tuning pipeline for Rajasthani dialect adaptation.
Wraps HuggingFace Seq2SeqTrainer with:
- Data collation for Whisper (audio → log-mel features)
- CER/WER metric computation during training
- Support for the 3-track hackathon strategy

Usage:
    trainer = ASRTrainer(model_name="indicwhisper-hindi-large-v2")
    trainer.train(
        train_data="data/processed/asr_corpus_train.jsonl",
        eval_data="data/processed/asr_corpus_val.jsonl",
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger

from src.asr.model import WhisperASR


@dataclass
class DataCollatorSpeechSeq2Seq:
    """
    Data collator for Whisper fine-tuning.

    Handles:
    - Padding input features (log-mel spectrograms) to batch max length
    - Padding labels (token IDs) and replacing pad tokens with -100
    """

    processor: Any
    decoder_start_token_id: int = 50258  # Whisper default

    def __call__(self, features: list[dict]) -> dict:
        # Extract input features and labels
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if it was prepended
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class ASRTrainer:
    """
    Trainer for fine-tuning Whisper on Rajasthani dialect audio.

    Implements the hackathon 3-track strategy:
    - Track 1: Zero-shot benchmark (no training needed)
    - Track 2: Joint "Rajasthani dialect" fine-tune (pool all 6 dialects)
    - Track 2b: Per-dialect fine-tune (Marwari, Bagri, etc.)
    """

    def __init__(
        self,
        model_name: str = "indicwhisper-hindi-large-v2",
        device: str = "auto",
    ):
        self.asr = WhisperASR(model_name=model_name, device=device if device != "auto" else None)
        logger.info(f"ASRTrainer initialized with {model_name}")

    def prepare_dataset(
        self,
        data_path: str | Path,
        max_samples: Optional[int] = None,
    ) -> list[dict]:
        """
        Load and prepare audio+text data for Whisper fine-tuning.

        Reads JSONL with fields: audio_path, text, dialect, sample_rate
        Returns list of dicts with input_features and labels.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            logger.warning(f"Data file not found: {data_path}")
            return []

        try:
            import librosa
        except ImportError:
            logger.error("librosa required: pip install librosa")
            raise

        processor = self.asr.processor
        records = []
        count = 0

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                audio_path = record.get("audio_path", "")
                text = record.get("text", "")
                if not audio_path or not text or not Path(audio_path).exists():
                    continue

                try:
                    # Load and resample audio to 16kHz
                    audio, sr = librosa.load(audio_path, sr=16000)

                    # Extract log-mel features
                    input_features = processor(
                        audio, sampling_rate=16000, return_tensors="np"
                    ).input_features[0]

                    # Tokenize text labels
                    labels = processor.tokenizer(text).input_ids

                    records.append({
                        "input_features": input_features,
                        "labels": labels,
                        "dialect": record.get("dialect", "unknown"),
                    })
                    count += 1

                    if count % 100 == 0:
                        logger.info(f"Prepared {count} samples...")

                except Exception as e:
                    logger.warning(f"Error processing {audio_path}: {e}")
                    continue

        logger.info(f"Prepared {len(records)} training samples from {data_path}")
        return records

    def train(
        self,
        train_data: str | Path,
        eval_data: Optional[str | Path] = None,
        output_dir: str = "checkpoints/asr_whisper",
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
    ) -> Path:
        """
        Fine-tune Whisper on dialect audio data.

        This is the main training entry point. Uses HuggingFace Seq2SeqTrainer
        under the hood.
        """
        try:
            from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
            from datasets import Dataset
        except ImportError:
            logger.error("Install: pip install transformers datasets")
            raise

        # Prepare model for fine-tuning
        self.asr.prepare_for_finetuning(freeze_encoder=True)

        # Load data
        logger.info("Preparing training data...")
        train_records = self.prepare_dataset(train_data, max_samples=max_train_samples)
        if not train_records:
            logger.error("No training data available. Aborting.")
            return Path(output_dir)

        train_dataset = Dataset.from_list(train_records)

        eval_dataset = None
        if eval_data and Path(eval_data).exists():
            logger.info("Preparing evaluation data...")
            eval_records = self.prepare_dataset(eval_data, max_samples=max_eval_samples)
            if eval_records:
                eval_dataset = Dataset.from_list(eval_records)

        # Data collator
        data_collator = DataCollatorSpeechSeq2Seq(
            processor=self.asr.processor,
            decoder_start_token_id=self.asr.model.config.decoder_start_token_id,
        )

        # Training arguments
        training_args_dict = self.asr.get_training_args(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
        )
        training_args = Seq2SeqTrainingArguments(**training_args_dict)

        # Metric computation
        def compute_metrics(pred):
            from src.evaluation.metrics import compute_cer, compute_wer

            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = self.asr.processor.tokenizer.pad_token_id

            pred_str = self.asr.processor.tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True
            )
            label_str = self.asr.processor.tokenizer.batch_decode(
                label_ids, skip_special_tokens=True
            )

            cers = [compute_cer(p, r) for p, r in zip(pred_str, label_str)]
            wers = [compute_wer(p, r) for p, r in zip(pred_str, label_str)]

            return {
                "cer": sum(cers) / max(len(cers), 1),
                "wer": sum(wers) / max(len(wers), 1),
            }

        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.asr.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics if eval_dataset else None,
            processing_class=self.asr.processor.feature_extractor,
        )

        # Train
        logger.info(f"Starting training: {num_epochs} epochs, {len(train_records)} samples")
        trainer.train()

        # Save final model
        output_path = Path(output_dir) / "final"
        self.asr.save_checkpoint(output_path)
        logger.success(f"Training complete. Model saved to {output_path}")

        return output_path

    def benchmark(
        self,
        test_data: str | Path,
        max_samples: Optional[int] = 200,
    ) -> dict:
        """
        Run zero-shot benchmark on test data.
        Track 1 of hackathon strategy.
        """
        test_path = Path(test_data)
        if not test_path.exists():
            logger.error(f"Test data not found: {test_path}")
            return {}

        audio_paths = []
        references = []

        with open(test_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    record = json.loads(line.strip())
                    ap = record.get("audio_path", "")
                    text = record.get("text", "")
                    if ap and text and Path(ap).exists():
                        audio_paths.append(ap)
                        references.append(text)
                except json.JSONDecodeError:
                    continue

        if not audio_paths:
            logger.warning("No valid test samples found")
            return {}

        return self.asr.benchmark_zero_shot(audio_paths, references)
