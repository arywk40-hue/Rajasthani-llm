"""
Corpus Builder

Orchestrates the data fusion and normalization pipeline.
It aggregates data from multiple sources (Vaani, BPCC, LDC-IL, Karya),
pushes all Devanagari text through the mandatory normalizer, filters out
poor-quality samples using the TextCleaner, deduplicates records, splits
into train/validation splits, and outputs clean unified datasets ready for
training the ASR, MT, and TTS models.

The LDC-IL corpus is treated as the GOLDEN evaluation set: it is written to
a dedicated output file and is NEVER mixed into train/val splits (roadmap §6).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from src.data.loaders import (
    AudioTextRecord,
    BPCCLoader,
    IndicTTSLoader,
    KaryaLoader,
    LDCILoader,
    ParallelTextRecord,
    VaaniLoader,
)
from src.preprocessing.normalizer import DevanagariNormalizer, create_normalizer_from_config
from src.preprocessing.text_cleaner import TextCleaner


@dataclass
class BuildStats:
    """Aggregate statistics for a corpus build run."""
    total_records: int = 0
    valid_records: int = 0
    rejected_records: int = 0
    duplicates_removed: int = 0
    train_records: int = 0
    val_records: int = 0
    per_source: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Total={self.total_records} valid={self.valid_records} "
            f"rejected={self.rejected_records} dupes_removed={self.duplicates_removed} "
            f"train={self.train_records} val={self.val_records} "
            f"sources={self.per_source}"
        )


class CorpusBuilder:
    """
    The central data fusion engine for the Rajasthani Dialect AI.

    Responsibilities:
    1. Ingest raw records from various dataset loaders.
    2. Apply non-negotiable Devanagari normalization to all text.
    3. Filter out invalid/noisy records using TextCleaner.
    4. Deduplicate records by content hash.
    5. Split into train/val splits (deterministic, seeded).
    6. Write processed, unified datasets (JSONL format) to the output directory.
    """

    def __init__(
        self,
        raw_data_dir: Path | str,
        output_dir: Path | str,
        normalizer_config_path: Optional[Path | str] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.val_ratio = val_ratio
        self.rng = random.Random(seed)

        # Initialize preprocessing tools
        config_path = Path(normalizer_config_path) if normalizer_config_path else None
        self.normalizer = create_normalizer_from_config(config_path)
        self.cleaner = TextCleaner(min_devanagari_ratio=0.3)

        logger.info(f"CorpusBuilder initialized. Outputs to {self.output_dir}")

    # ─── Record processing helpers ────────────────────────────────────────────

    def process_text(self, text: str) -> Optional[str]:
        """
        Clean and normalize text. Returns None if text fails quality checks.
        """
        # Step 1: Clean (removes URLs, noise tags, normalizes whitespace)
        cleaned = self.cleaner.clean(text)

        # Step 2: Assess quality (must be viable for training)
        score = self.cleaner.assess_quality(cleaned)
        if not score.is_viable:
            return None

        # Step 3: Strict Devanagari Normalization (NFC, Nukta, etc.)
        normalized = self.normalizer.normalize(cleaned)
        return normalized

    @staticmethod
    def _content_hash(record: dict) -> str:
        """Stable content fingerprint for deduplication."""
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _split_files(self, stem: str) -> tuple[Path, Path]:
        return self.output_dir / f"{stem}_train.jsonl", self.output_dir / f"{stem}_val.jsonl"

    def _write_splits(
        self,
        records: list[dict],
        stem: str,
        stats: BuildStats,
    ) -> tuple[Path, Path]:
        """
        Deterministically split a list of record dicts into train/val JSONL
        and write them to disk.
        """
        train_path, val_path = self._split_files(stem)
        seen = set()
        train_records: list[dict] = []
        val_records: list[dict] = []

        for record in records:
            h = self._content_hash(record)
            if h in seen:
                stats.duplicates_removed += 1
                continue
            seen.add(h)
            if self.rng.random() < self.val_ratio:
                val_records.append(record)
            else:
                train_records.append(record)

        with open(train_path, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(val_path, "w", encoding="utf-8") as f:
            for r in val_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        stats.train_records = len(train_records)
        stats.val_records = len(val_records)
        logger.info(f"Split '{stem}': train={len(train_records)} val={len(val_records)}")
        return train_path, val_path

    # ─── Corpus builders ──────────────────────────────────────────────────────

    def build_asr_corpus(self, dialects: list[str]) -> tuple[Path, Path]:
        """
        Builds the unified acoustic corpus for ASR training.
        Fuses VAANI (for dialects) and Karya (broad augmentation).
        Returns (train_path, val_path).
        """
        stem = "asr_corpus"
        stats = BuildStats()
        records: list[dict] = []

        vaani_loader = VaaniLoader(self.raw_data_dir / "vaani")
        karya_loader = KaryaLoader(self.raw_data_dir / "karya")

        logger.info(f"Building ASR corpus for dialects: {dialects}")

        for dialect in dialects:
            for record in vaani_loader.iter_records(dialect=dialect):
                stats.total_records += 1
                proc_text = self.process_text(record.text)
                if not proc_text:
                    stats.rejected_records += 1
                    continue
                record.text = proc_text
                records.append(record.__dict__)
                stats.valid_records += 1
                stats.per_source[record.source_dataset] = stats.per_source.get(record.source_dataset, 0) + 1

        for record in karya_loader.iter_records():
            stats.total_records += 1
            proc_text = self.process_text(record.text)
            if not proc_text:
                stats.rejected_records += 1
                continue
            record.text = proc_text
            records.append(record.__dict__)
            stats.valid_records += 1
            stats.per_source[record.source_dataset] = stats.per_source.get(record.source_dataset, 0) + 1

        train_path, val_path = self._write_splits(records, stem, stats)
        self._log_stats("ASR", stats)
        return train_path, val_path

    def build_mt_corpus(self, src_lang: str, tgt_lang: str) -> tuple[Path, Path]:
        """
        Builds the parallel text corpus for MT training.
        Returns (train_path, val_path).
        """
        stem = f"mt_corpus_{src_lang}_{tgt_lang}"
        stats = BuildStats()
        records: list[dict] = []

        bpcc_loader = BPCCLoader(self.raw_data_dir / "bpcc")

        logger.info(f"Building MT corpus for {src_lang}-{tgt_lang}")

        for record in bpcc_loader.iter_records(src_lang, tgt_lang):
            stats.total_records += 1

            # Normalize both source and target (if they are Indic languages)
            proc_src = (
                self.process_text(record.source_text) if src_lang != "en" else self.cleaner.clean(record.source_text)
            )
            proc_tgt = (
                self.process_text(record.target_text) if tgt_lang != "en" else self.cleaner.clean(record.target_text)
            )

            # For MT, both sides must be viable
            if not proc_src or not proc_tgt:
                stats.rejected_records += 1
                continue
            record.source_text = proc_src
            record.target_text = proc_tgt
            records.append(record.__dict__)
            stats.valid_records += 1
            stats.per_source[record.source_dataset] = stats.per_source.get(record.source_dataset, 0) + 1

        train_path, val_path = self._write_splits(records, stem, stats)
        self._log_stats("MT", stats)
        return train_path, val_path

    def build_golden_eval_corpus(self, src_lang: str = "rj", tgt_lang: str = "hi") -> Path:
        """
        Builds the LDC-IL GOLDEN evaluation corpus.
        This set is emitted in its entirety to 'ldcil_golden.jsonl' and is
        NEVER placed in any train/val split — it is held out for evaluation.
        """
        output_path = self.output_dir / "ldcil_golden.jsonl"
        stats = BuildStats()

        ldc_loader = LDCILoader(self.raw_data_dir / "ldcil")

        count = 0
        with open(output_path, "w", encoding="utf-8") as out:
            for record in ldc_loader.iter_records(src_lang, tgt_lang):
                stats.total_records += 1
                proc_src = self.process_text(record.source_text)
                proc_tgt = self.process_text(record.target_text)
                if not proc_src or not proc_tgt:
                    stats.rejected_records += 1
                    continue
                record.source_text = proc_src
                record.target_text = proc_tgt
                out.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
                count += 1
                stats.valid_records += 1

        logger.info(f"Golden eval corpus built: {count}/{stats.total_records} valid records -> {output_path}")
        return output_path

    def build_tts_corpus(self) -> tuple[Path, Path]:
        """
        Builds the high-fidelity phonetic acoustic corpus for TTS.
        Returns (train_path, val_path).
        """
        stem = "tts_corpus"
        stats = BuildStats()
        records: list[dict] = []

        tts_loader = IndicTTSLoader(self.raw_data_dir / "indic_tts")

        logger.info("Building TTS corpus")

        for record in tts_loader.iter_records("rajasthani"):
            stats.total_records += 1
            proc_text = self.process_text(record.text)
            if not proc_text:
                stats.rejected_records += 1
                continue
            record.text = proc_text
            records.append(record.__dict__)
            stats.valid_records += 1
            stats.per_source[record.source_dataset] = stats.per_source.get(record.source_dataset, 0) + 1

        train_path, val_path = self._write_splits(records, stem, stats)
        self._log_stats("TTS", stats)
        return train_path, val_path

    def _log_stats(self, component: str, stats: BuildStats) -> None:
        logger.info(f"{component} corpus stats: {stats.summary()}")
        if self.normalizer.stats:
            logger.info(f"Normalization Stats: {self.normalizer.stats.summary()}")


if __name__ == "__main__":
    # Example execution
    builder = CorpusBuilder(
        raw_data_dir="data/raw",
        output_dir="data/processed",
        normalizer_config_path="config/base.yaml",
    )
    builder.build_asr_corpus(["marwari", "mewari", "dhundhari", "hadoti", "mewati", "bagri"])
    builder.build_mt_corpus("en", "hi")
    builder.build_golden_eval_corpus("rj", "hi")
    builder.build_tts_corpus()