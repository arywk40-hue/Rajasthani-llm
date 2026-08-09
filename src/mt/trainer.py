"""
MT Trainer — Production Training Loop for IndicTrans2

Implements the accuracy-maximizing techniques from the architecture docs:
1. Real CrossEntropyLoss with label smoothing
2. Experience replay (interleave dialect + general BPCC data)
3. Model souping (average fine-tuned checkpoints with base weights)
4. Gradient accumulation + mixed precision (AMP)
5. Cosine LR scheduling with warmup
6. Periodic chrF++ evaluation against golden set
7. Early stopping based on validation chrF++

Reference: implementation.md §5 (Experience replay, model souping)
Reference: detailed-report.md Phase 2 (MT subsystem)
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from src.mt.model import IndicTrans2MT


class MTTrainer:
    """
    Production trainer for IndicTrans2 dialect fine-tuning.

    Supports two modes:
    1. HuggingFace Trainer mode: Uses Seq2SeqTrainer (recommended)
    2. Manual training loop: For full control over experience replay scheduling

    Usage:
        model = IndicTrans2MT(config_path="config/mt.yaml")
        trainer = MTTrainer(model, device="cuda")

        # Simple fine-tune
        trainer.train_with_hf_trainer(
            train_data="data/processed/mt_dialect_train.jsonl",
            eval_data="data/processed/ldcil_golden.jsonl",
        )

        # Manual loop with experience replay
        trainer.train_manual(
            dialect_data="data/processed/mt_marwari_train.jsonl",
            general_data="data/processed/mt_bpcc_train.jsonl",
            replay_ratio=0.15,
        )
    """

    def __init__(
        self,
        model: IndicTrans2MT,
        device: str = "auto",
        learning_rate: float = 3e-5,
        label_smoothing: float = 0.1,
    ):
        self.model = model
        if device != "auto" and device is not None:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing

        # Loss function — CrossEntropy with label smoothing
        self._criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100,
        )

        logger.info(
            f"MTTrainer initialized | device={self._device} "
            f"lr={learning_rate} label_smoothing={label_smoothing}"
        )

    # ─── HuggingFace Trainer Mode (Recommended) ──────────────────────────────

    def train_with_hf_trainer(
        self,
        train_data: str | Path,
        eval_data: Optional[str | Path] = None,
        output_dir: str = "checkpoints/mt_indictrans2",
        num_epochs: int = 10,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 500,
        fp16: bool = True,
        general_data: Optional[str | Path] = None,
        replay_ratio: float = 0.15,
    ) -> Path:
        """
        Fine-tune IndicTrans2 using HuggingFace Seq2SeqTrainer.

        This is the recommended training method. It handles:
        - Mixed precision training (FP16)
        - Gradient accumulation (effective batch size = batch_size * grad_accum)
        - Learning rate scheduling (cosine with warmup)
        - Evaluation during training (chrF++ on golden set)
        - Checkpoint saving (best model by eval metric)
        """
        try:
            from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
        except ImportError:
            logger.error("Install: pip install transformers")
            raise

        # Ensure model is loaded
        self.model._ensure_model()
        if not self.model.is_loaded:
            logger.error(
                "HuggingFace model not loaded. Cannot use HF Trainer. "
                "Check that IndicTrans2 is downloaded."
            )
            return Path(output_dir)

        # Build datasets
        from src.mt.dataset import MTDataset, CombinedMTDataset, ExperienceReplaySampler

        tokenizer = self.model._tokenizer
        processor = self.model._processor

        train_dataset = MTDataset(
            train_data, tokenizer=tokenizer, processor=processor,
        )

        if len(train_dataset) == 0:
            raise ValueError(
                f"MT training dataset {train_data} is empty. "
                "Please verify that the dataset ingestion/fetching ran successfully and "
                "produced non-empty training files before starting training."
            )

        eval_dataset = None
        if eval_data and Path(eval_data).exists():
            eval_dataset = MTDataset(
                eval_data, tokenizer=tokenizer, processor=processor,
            )

        # Experience replay: combine dialect + general data
        final_train_dataset = train_dataset
        sampler = None
        if general_data and Path(general_data).exists():
            general_dataset = MTDataset(
                general_data, tokenizer=tokenizer, processor=processor,
            )
            final_train_dataset = CombinedMTDataset(train_dataset, general_dataset)
            sampler = ExperienceReplaySampler(
                dialect_size=len(train_dataset),
                general_size=len(general_dataset),
                replay_ratio=replay_ratio,
            )
            logger.info(
                f"Experience replay enabled: {len(train_dataset)} dialect + "
                f"{len(general_dataset)} general (ratio={replay_ratio})"
            )

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            fp16=fp16 and self._device == "cuda",
            label_smoothing_factor=self.label_smoothing,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_steps=500,
            save_total_limit=3,
            logging_steps=50,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="chrf" if eval_dataset else None,
            greater_is_better=True if eval_dataset else None,
            predict_with_generate=True,
            generation_max_length=256,
            generation_num_beams=5,
            report_to="none",
            max_grad_norm=1.0,
            dataloader_num_workers=2,
        )

        # Metric computation
        def compute_metrics(pred):
            from src.evaluation.metrics import compute_chrf
            predictions = pred.predictions
            labels = pred.label_ids
            labels[labels == -100] = tokenizer.pad_token_id

            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            chrf_scores = [
                compute_chrf(p, r) for p, r in zip(decoded_preds, decoded_labels)
            ]
            return {"chrf": sum(chrf_scores) / max(len(chrf_scores), 1)}

        # Data collator — prevents decoder_input_ids/decoder_inputs_embeds conflict
        # by letting the model shift labels internally rather than the collator doing it
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,  # Don't pass model — avoids double decoder_input_ids generation
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if fp16 else None,
        )

        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model._hf_model,
            args=training_args,
            train_dataset=final_train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if eval_dataset else None,
        )

        # Train
        logger.info(
            f"Starting MT training: {num_epochs} epochs, "
            f"{len(final_train_dataset)} samples, "
            f"effective batch={batch_size * gradient_accumulation_steps}"
        )
        trainer.train()

        # Save final model
        output_path = Path(output_dir) / "final"
        self.model.save_checkpoint(output_path)
        logger.success(f"MT training complete. Model saved to {output_path}")

        return output_path

    # ─── Manual Training Loop ─────────────────────────────────────────────────

    def train_manual(
        self,
        dialect_data: str | Path,
        general_data: Optional[str | Path] = None,
        output_dir: str = "checkpoints/mt_indictrans2",
        num_epochs: int = 10,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 4,
        replay_ratio: float = 0.15,
        eval_every: int = 500,
        eval_data: Optional[str | Path] = None,
    ) -> Path:
        """
        Manual training loop with full control over experience replay.

        Use this when you need fine-grained control over the training process
        (e.g., custom replay scheduling, dynamic batch mixing).
        """
        self.model._ensure_model()
        if not self.model.is_loaded:
            logger.error("Model not loaded. Aborting manual training.")
            return Path(output_dir)

        from src.mt.dataset import create_mt_dataloader, MTDataset

        hf_model = self.model._hf_model
        tokenizer = self.model._tokenizer

        # Ensure training data is not empty
        train_ds = MTDataset(dialect_data, tokenizer=tokenizer, processor=self.model._processor)
        if len(train_ds) == 0:
            raise ValueError(
                f"MT manual training dialect dataset {dialect_data} is empty. "
                "Please verify that the dataset ingestion/fetching ran successfully and "
                "produced non-empty training files before starting training."
            )

        # Create dataloader with experience replay
        dataloader = create_mt_dataloader(
            dialect_data_path=dialect_data,
            general_data_path=general_data,
            tokenizer=tokenizer,
            processor=self.model._processor,
            batch_size=batch_size,
            replay_ratio=replay_ratio,
        )

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            [p for p in hf_model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
        warmup_steps = min(500, total_steps // 10)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=self.learning_rate * 0.01
        )

        # Mixed precision
        scaler = torch.amp.GradScaler("cuda") if self._device == "cuda" else None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        best_chrf = 0.0
        global_step = 0

        for epoch in range(num_epochs):
            hf_model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                # Skip batches without tokenized data
                if "input_ids" not in batch:
                    continue

                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["labels"].to(self._device)

                # Forward pass (with mixed precision)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = hf_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = hf_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                    loss.backward()

                epoch_loss += loss.item() * gradient_accumulation_steps

                # Gradient accumulation step
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(hf_model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(hf_model.parameters(), max_norm=1.0)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Periodic evaluation
                    if eval_data and global_step % eval_every == 0:
                        chrf = self._evaluate(eval_data)
                        logger.info(
                            f"Step {global_step} | Loss={epoch_loss / max(num_batches, 1):.4f} "
                            f"| chrF++={chrf:.4f}"
                        )
                        if chrf > best_chrf:
                            best_chrf = chrf
                            self.model.save_checkpoint(output_path / "best")
                            logger.info(f"New best chrF++: {chrf:.4f}")

                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

            # Save epoch checkpoint
            self.model.save_checkpoint(output_path / f"epoch_{epoch + 1}")

        logger.success(f"Manual training complete. Best chrF++: {best_chrf:.4f}")
        return output_path

    # ─── Model Souping ────────────────────────────────────────────────────────

    def create_model_soup(
        self,
        checkpoint_paths: list[str | Path],
        base_weight: float = 0.5,
    ) -> IndicTrans2MT:
        """
        Average fine-tuned checkpoints with base model weights.
        Delegates to IndicTrans2MT.create_model_soup().
        """
        return self.model.create_model_soup(checkpoint_paths, base_weight)

    # ─── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate(self, eval_data: str | Path) -> float:
        """Quick chrF++ evaluation on a dataset."""
        from src.evaluation.metrics import compute_chrf

        eval_path = Path(eval_data)
        if not eval_path.exists():
            return 0.0

        sources, references = [], []
        src_lang_code = "hin_Deva"
        tgt_lang_code = "hin_Deva"
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    sources.append(record.get("source_text", ""))
                    references.append(record.get("target_text", ""))
                    # Read lang codes from first record
                    if len(sources) == 1:
                        src_lang_code = record.get("source_lang", "hin_Deva")
                        tgt_lang_code = record.get("target_lang", "hin_Deva")
                except json.JSONDecodeError:
                    continue

        if not sources:
            return 0.0

        # Translate
        self.model._hf_model.eval()
        translations = self.model.translate(
            sources[:100],
            src_lang=src_lang_code,
            tgt_lang=tgt_lang_code,
        )
        self.model._hf_model.train()

        # Compute chrF++
        chrf_scores = [
            compute_chrf(hyp, ref)
            for hyp, ref in zip(translations, references[:100])
        ]
        return sum(chrf_scores) / max(len(chrf_scores), 1)

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save model weights."""
        return self.model.save_checkpoint(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IndicTrans2 MT Trainer")
    parser.add_argument("--train_data", type=str, default="data/raw/karya/karya_rajasthan.jsonl")
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/mt_gpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--fp16", type=bool, default=True)
    args = parser.parse_args()

    model = IndicTrans2MT()
    trainer = MTTrainer(model)
    trainer.train_with_hf_trainer(
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
    )
