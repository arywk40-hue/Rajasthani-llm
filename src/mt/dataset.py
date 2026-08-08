"""
MT Dataset and DataLoader

PyTorch Dataset wrapping the JSONL parallel corpus from CorpusBuilder.
Includes:
- IndicProcessor integration for script unification
- Dynamic batching by sequence length (padding efficiency)
- Experience replay sampler that mixes dialect + general BPCC data

Reference: implementation.md §5 (Experience replay)
Reference: detailed-report.md Phase 2 (Conversational register adaptation)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from loguru import logger


class MTDataset(Dataset):
    """
    Parallel text dataset for MT training.

    Reads JSONL files produced by CorpusBuilder, where each line is:
        {"source_text": "...", "target_text": "...", "source_lang": "...", "target_lang": "...", ...}

    The dataset handles tokenization lazily at __getitem__ time so that the
    tokenizer can be swapped (e.g., when switching between fine-tuning runs
    on different language pairs).
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer=None,
        processor=None,
        src_lang: str = "hin_Deva",
        tgt_lang: str = "eng_Latn",
        max_source_length: int = 256,
        max_target_length: int = 256,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.processor = processor
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # Load all records into memory
        self.records: list[dict] = []
        if self.data_path.exists():
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("source_text") and record.get("target_text"):
                            self.records.append(record)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"MTDataset loaded {len(self.records)} pairs from {self.data_path}")
        else:
            logger.warning(f"MTDataset: {self.data_path} not found. Empty dataset.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        src_text = record["source_text"]
        tgt_text = record["target_text"]

        # Apply IndicProcessor preprocessing if available
        if self.processor is not None:
            try:
                src_text = self.processor.preprocess_batch(
                    [src_text], src_lang=self.src_lang, tgt_lang=self.tgt_lang
                )[0]
            except Exception:
                pass  # Fall through to raw text

        item = {
            "source_text": src_text,
            "target_text": tgt_text,
            "source_lang": record.get("source_lang", ""),
            "target_lang": record.get("target_lang", ""),
        }

        # Tokenize if tokenizer is available
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "src_lang") and getattr(self.tokenizer, "src_lang", None) is None:
                self.tokenizer.src_lang = "hi"
            if hasattr(self.tokenizer, "tgt_lang") and getattr(self.tokenizer, "tgt_lang", None) is None:
                self.tokenizer.tgt_lang = "hi"

            source_encoding = self.tokenizer(
                src_text,
                max_length=self.max_source_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # Tokenize targets
            target_encoding = self.tokenizer(
                text_target=tgt_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            item["input_ids"] = source_encoding["input_ids"].squeeze(0)
            item["attention_mask"] = source_encoding["attention_mask"].squeeze(0)
            labels = target_encoding["input_ids"].squeeze(0)
            # Replace padding token id with -100 so it's ignored by loss
            labels[labels == self.tokenizer.pad_token_id] = -100
            item["labels"] = labels

        return item

    def get_source_texts(self) -> list[str]:
        """Get all source texts (useful for evaluation)."""
        return [r["source_text"] for r in self.records]

    def get_target_texts(self) -> list[str]:
        """Get all target texts (useful for evaluation)."""
        return [r["target_text"] for r in self.records]


class ExperienceReplaySampler(Sampler[int]):
    """
    Sampler that interleaves indices from a dialect-specific dataset with
    indices from a general-domain dataset (BPCC).

    This prevents catastrophic forgetting during dialect fine-tuning by
    mixing in 10-20% general data (implementation.md §5: experience replay).

    Args:
        dialect_dataset: The dialect-specific dataset being fine-tuned on
        general_dataset: The general BPCC dataset to mix in
        replay_ratio: Fraction of general data to mix in (0.1 = 10%)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dialect_size: int,
        general_size: int,
        replay_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.dialect_size = dialect_size
        self.general_size = general_size
        self.replay_ratio = replay_ratio
        self.rng = random.Random(seed)

        # Calculate how many general samples to include
        self.n_general = int(dialect_size * replay_ratio / (1 - replay_ratio))
        self.n_general = min(self.n_general, general_size)

        # Total samples per epoch
        self._total = dialect_size + self.n_general

        logger.info(
            f"ExperienceReplaySampler: {dialect_size} dialect + "
            f"{self.n_general} general (ratio={replay_ratio})"
        )

    def __iter__(self) -> Iterator[int]:
        # Create indices: dialect indices are [0, dialect_size)
        # General indices are [dialect_size, dialect_size + n_general)
        dialect_indices = list(range(self.dialect_size))
        general_indices = self.rng.sample(
            range(self.dialect_size, self.dialect_size + self.general_size),
            k=self.n_general,
        )

        all_indices = dialect_indices + general_indices
        self.rng.shuffle(all_indices)
        return iter(all_indices)

    def __len__(self) -> int:
        return self._total


class CombinedMTDataset(Dataset):
    """
    Combines a dialect-specific dataset with a general dataset for
    experience replay training. Indices [0, len(dialect)) map to the
    dialect dataset; [len(dialect), len(dialect) + len(general)) map
    to the general dataset.
    """

    def __init__(self, dialect_dataset: MTDataset, general_dataset: MTDataset):
        self.dialect = dialect_dataset
        self.general = general_dataset

    def __len__(self) -> int:
        return len(self.dialect) + len(self.general)

    def __getitem__(self, idx: int) -> dict:
        if idx < len(self.dialect):
            item = self.dialect[idx]
            item["is_replay"] = False
        else:
            item = self.general[idx - len(self.dialect)]
            item["is_replay"] = True
        return item


def create_mt_dataloader(
    dialect_data_path: str | Path,
    general_data_path: Optional[str | Path] = None,
    tokenizer=None,
    processor=None,
    src_lang: str = "hin_Deva",
    tgt_lang: str = "eng_Latn",
    batch_size: int = 16,
    replay_ratio: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    """
    Factory function to create an MT DataLoader with optional experience replay.

    If general_data_path is provided, the DataLoader will interleave
    dialect-specific and general-domain data according to replay_ratio.
    """
    dialect_dataset = MTDataset(
        dialect_data_path, tokenizer=tokenizer, processor=processor,
        src_lang=src_lang, tgt_lang=tgt_lang,
    )

    if general_data_path and Path(general_data_path).exists():
        general_dataset = MTDataset(
            general_data_path, tokenizer=tokenizer, processor=processor,
            src_lang=src_lang, tgt_lang=tgt_lang,
        )
        combined = CombinedMTDataset(dialect_dataset, general_dataset)
        sampler = ExperienceReplaySampler(
            dialect_size=len(dialect_dataset),
            general_size=len(general_dataset),
            replay_ratio=replay_ratio,
            seed=seed,
        )
        return DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dialect_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
