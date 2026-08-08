"""
Dataset Loaders for the Rajasthani Dialect AI

Implements specialized data loaders for the fundamental datasets:
1. VaaniLoader: ARTPARK-IISc VAANI (Audio/Text Parquet)
2. BPCCLoader: Bharat Parallel Corpus Collection (Parallel Text)
3. IndicTTSLoader: IndicTTS Database (Audio/Phonetic Text)
4. KaryaLoader: Speech-rj-hi (Karya) (Audio/Text Parquet)

All loaders are responsible for safely reading raw data, enforcing schema
expectations, and yielding standardized records for the CorpusBuilder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from loguru import logger


@dataclass
class AudioTextRecord:
    """Standardized record for speech data."""
    audio_path: str
    text: str
    dialect: str
    speaker_id: Optional[str] = None
    duration_sec: Optional[float] = None
    source_dataset: str = ""


@dataclass
class ParallelTextRecord:
    """Standardized record for translation data."""
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    source_dataset: str = ""


class VaaniLoader:
    """
    Loader for the ARTPARK-IISc VAANI dataset.
    Reads Parquet files containing dialectal audio metadata and transcripts.
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.dataset_name = "vaani"

    def iter_records(self, dialect: Optional[str] = None) -> Iterator[AudioTextRecord]:
        """
        Iterate through VAANI records, optionally filtered by dialect.
        Yields normalized AudioTextRecord objects.
        """
        parquet_files = list(self.data_dir.glob("**/*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in VAANI dir: {self.data_dir}")
            return

        logger.info(f"Loading VAANI from {len(parquet_files)} parquet files.")
        for pfile in parquet_files:
            try:
                # Read parquet using pandas (pyarrow engine under the hood)
                df = pd.read_parquet(pfile)
                # Standard Vaani schema often has 'audio', 'text', 'district'/'language'
                for _, row in df.iterrows():
                    record_dialect = str(row.get("language", "unknown")).lower()
                    if dialect and dialect.lower() != record_dialect:
                        continue
                    
                    audio_path = str(row.get("audio", ""))
                    text = str(row.get("text", ""))
                    speaker = str(row.get("speaker_id", ""))
                    
                    if not text or not audio_path:
                        continue
                        
                    yield AudioTextRecord(
                        audio_path=audio_path,
                        text=text,
                        dialect=record_dialect,
                        speaker_id=speaker,
                        source_dataset=self.dataset_name,
                    )
            except Exception as e:
                logger.error(f"Error reading VAANI parquet {pfile}: {e}")


class BPCCLoader:
    """
    Loader for the Bharat Parallel Corpus Collection (IndicTrans2).
    Reads bilingual TSV or parallel text files.
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.dataset_name = "bpcc"

    def iter_records(self, src_lang: str, tgt_lang: str) -> Iterator[ParallelTextRecord]:
        """
        Iterate through parallel sentences.
        Expects directories named like 'en-hi' containing 'train.src' and 'train.tgt'.
        """
        pair_dir = self.data_dir / f"{src_lang}-{tgt_lang}"
        if not pair_dir.exists():
            pair_dir = self.data_dir / f"{tgt_lang}-{src_lang}"
            if not pair_dir.exists():
                logger.warning(f"No BPCC directory found for pair {src_lang}-{tgt_lang}")
                return

        src_files = sorted(pair_dir.glob("*.src"))
        tgt_files = sorted(pair_dir.glob("*.tgt"))

        if not src_files or len(src_files) != len(tgt_files):
            logger.warning(f"Mismatched or missing src/tgt files in {pair_dir}")
            return

        for src_file, tgt_file in zip(src_files, tgt_files):
            with open(src_file, "r", encoding="utf-8") as fs, open(tgt_file, "r", encoding="utf-8") as ft:
                for s_line, t_line in zip(fs, ft):
                    s_text = s_line.strip()
                    t_text = t_line.strip()
                    if s_text and t_text:
                        yield ParallelTextRecord(
                            source_text=s_text,
                            target_text=t_text,
                            source_lang=src_lang,
                            target_lang=tgt_lang,
                            source_dataset=self.dataset_name,
                        )


class LDCILoader:
    """
    Loader for the LDC-IL Rajasthani Corpus.

    Per the roadmap (Section 4.1 / 6), this ~5,332-sentence parallel corpus is
    the GOLDEN MT evaluation set. It must be kept strictly out of any training
    run to avoid data leakage. This loader consumes a tab-separated parallel
    file with the schema: `source\\ttarget` on each line.

    Layout convention (mirrors the corpus inventory):
        data/raw/ldcil/{src_tgt}.tsv

    Args:
        data_dir: Directory containing the LDC-IL parallel TSV files.
        src_lang: Source language code (default 'rj' — Rajasthani).
        tgt_lang: Target language code (default 'hi' — Hindi).
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.dataset_name = "ldcil"

    def iter_records(self, src_lang: str = "rj", tgt_lang: str = "hi") -> Iterator[ParallelTextRecord]:
        """Yield golden parallel sentence records from the LDC-IL corpus."""
        tsv_file = self.data_dir / f"{src_lang}-{tgt_lang}.tsv"
        if not tsv_file.exists():
            logger.warning(f"LDC-IL golden TSV not found: {tsv_file}")
            return

        logger.info(f"Loading LDC-IL golden corpus from {tsv_file}")
        with open(tsv_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                src, tgt = parts[0].strip(), parts[1].strip()
                if src and tgt:
                    yield ParallelTextRecord(
                        source_text=src,
                        target_text=tgt,
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        source_dataset=self.dataset_name,
                    )

    def load_golden_pairs(self, src_lang: str = "rj", tgt_lang: str = "hi") -> list[tuple[str, str]]:
        """Return the full golden evaluation set as a list of (source, reference) tuples."""
        return [(r.source_text, r.target_text) for r in self.iter_records(src_lang, tgt_lang)]


class IndicTTSLoader:
    """
    Loader for the IndicTTS database.
    Expects LJSpeech format (metadata.csv with audio path and text).
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.dataset_name = "indic_tts"

    def iter_records(self, language: str = "rajasthani") -> Iterator[AudioTextRecord]:
        """Iterate through IndicTTS records for a given macro-language."""
        lang_dir = self.data_dir / language
        metadata_file = lang_dir / "metadata.csv"
        
        if not metadata_file.exists():
            logger.warning(f"IndicTTS metadata not found: {metadata_file}")
            return
            
        try:
            df = pd.read_csv(metadata_file, sep="|", header=None, names=["id", "text", "text_normalized"])
            for _, row in df.iterrows():
                audio_id = str(row["id"])
                text = str(row["text"])
                audio_path = str(lang_dir / "wavs" / f"{audio_id}.wav")
                
                yield AudioTextRecord(
                    audio_path=audio_path,
                    text=text,
                    dialect=language,  # IndicTTS groups all dialects under 'rajasthani'
                    source_dataset=self.dataset_name,
                )
        except Exception as e:
            logger.error(f"Error reading IndicTTS metadata {metadata_file}: {e}")


class KaryaLoader:
    """
    Loader for the Speech-rj-hi (Karya) dataset.
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.dataset_name = "karya"

    def iter_records(self) -> Iterator[AudioTextRecord]:
        """Iterate through Karya parquet records."""
        parquet_files = list(self.data_dir.glob("**/*.parquet"))
        for pfile in parquet_files:
            try:
                df = pd.read_parquet(pfile)
                for _, row in df.iterrows():
                    audio_path = str(row.get("audio", ""))
                    text = str(row.get("text", ""))
                    
                    if not text or not audio_path:
                        continue
                        
                    yield AudioTextRecord(
                        audio_path=audio_path,
                        text=text,
                        dialect="rajasthani",  # Broad categorization
                        source_dataset=self.dataset_name,
                    )
            except Exception as e:
                logger.error(f"Error reading Karya parquet {pfile}: {e}")
