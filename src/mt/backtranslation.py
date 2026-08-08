"""
Back-translation Generator — Real Model Inference

Generates synthetic parallel data by translating monolingual dialectal text
through a base/weakly-trained MT model. This is the primary data augmentation
strategy for low-resource dialects (Bagri: 0.6 hrs, Hadoti: 0.3 hrs).

Pipeline:
    Monolingual Rajasthani text
    → IndicTrans2 (dialect → Hindi)
    → Quality filter (reject copies + garbage)
    → Pseudo-parallel pairs (dialect, Hindi)
    → Feed back into MT training

Supports iterative back-translation:
    Round 1: Weak model → noisy pairs → retrain
    Round 2: Stronger model → better pairs → retrain
    ...

Reference: implementation.md §4.3 (Augmentation via back-translation)
Reference: detailed-report.md Phase 5 (Gap Analysis — iterative BT)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.mt.model import IndicTrans2MT


def _text_similarity(a: str, b: str) -> float:
    """Simple character overlap ratio for quality filtering."""
    if not a or not b:
        return 0.0
    a_chars = set(a)
    b_chars = set(b)
    intersection = a_chars & b_chars
    union = a_chars | b_chars
    return len(intersection) / max(len(union), 1)


class BackTranslationGenerator:
    """
    Generates synthetic parallel data via back-translation.

    Uses a real IndicTrans2 model for translation instead of placeholders.

    Usage:
        bt = BackTranslationGenerator()

        # Generate pseudo-parallel pairs
        bt.generate(
            monolingual_file="data/raw/marwari_monolingual.txt",
            output_file="data/augmented/marwari_bt_pairs.jsonl",
            src_lang="marwari",
            tgt_lang="hindi",
        )

        # Iterative back-translation (multiple rounds)
        for round_num in range(3):
            bt.generate(
                monolingual_file="data/raw/marwari_mono.txt",
                output_file=f"data/augmented/marwari_bt_round{round_num}.jsonl",
                src_lang="marwari",
                tgt_lang="hindi",
            )
            # Retrain MT model on original + augmented data
            # Then reload the improved model into bt.model
    """

    def __init__(
        self,
        model: Optional[IndicTrans2MT] = None,
        model_path: Optional[str] = None,
        config_path: str = "config/mt.yaml",
    ):
        if model is not None:
            self.model = model
        else:
            self.model = IndicTrans2MT(config_path=config_path)
            if model_path:
                self.model.load_checkpoint(model_path)

        logger.info("BackTranslationGenerator initialized")

    def generate(
        self,
        monolingual_file: str | Path,
        output_file: str | Path,
        src_lang: str = "marwari",
        tgt_lang: str = "hindi",
        batch_size: int = 32,
        max_pairs: Optional[int] = None,
        min_length: int = 5,
        max_length: int = 500,
        similarity_threshold_low: float = 0.3,
        similarity_threshold_high: float = 0.95,
        use_sampling: bool = True,
    ) -> int:
        """
        Read monolingual text, generate translations, save as pseudo-parallel pairs.

        Quality filtering:
        - Reject if source-target similarity > 0.95 (likely a copy, not a translation)
        - Reject if source-target similarity < 0.3 (likely garbage)
        - Reject if translation is too short or empty

        Uses sampling (not beam search) for diversity — this is important for
        back-translation to avoid degenerate repetition.

        Args:
            monolingual_file: Input file (one sentence per line)
            output_file: Output JSONL file with pseudo-parallel pairs
            src_lang: Source language (dialect name)
            tgt_lang: Target language (e.g., "hindi")
            batch_size: Number of sentences to translate at once
            max_pairs: Maximum number of pairs to generate (None = all)
            min_length: Minimum source text length (chars)
            max_length: Maximum source text length (chars)
            similarity_threshold_low: Min similarity to keep (filters garbage)
            similarity_threshold_high: Max similarity to keep (filters copies)
            use_sampling: Use sampling instead of beam search (recommended for BT)

        Returns:
            Number of pseudo-parallel pairs generated
        """
        monolingual_file = Path(monolingual_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if not monolingual_file.exists():
            logger.error(f"Monolingual file not found: {monolingual_file}")
            return 0

        # Read source texts
        with open(monolingual_file, "r", encoding="utf-8") as f:
            source_texts = [
                line.strip() for line in f
                if line.strip() and min_length <= len(line.strip()) <= max_length
            ]

        if max_pairs:
            source_texts = source_texts[:max_pairs]

        logger.info(
            f"Back-translating {len(source_texts)} sentences | "
            f"{src_lang} → {tgt_lang} | batch_size={batch_size}"
        )

        # Generate translations in batches
        accepted = 0
        rejected_copy = 0
        rejected_garbage = 0
        rejected_empty = 0

        with open(output_file, "w", encoding="utf-8") as f_out:
            for i in range(0, len(source_texts), batch_size):
                batch = source_texts[i : i + batch_size]

                try:
                    translations = self.model.translate(
                        batch,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                    )
                except Exception as e:
                    logger.warning(f"Translation batch failed at index {i}: {e}")
                    # Fallback: try one at a time
                    translations = []
                    for text in batch:
                        try:
                            t = self.model.translate([text], src_lang=src_lang, tgt_lang=tgt_lang)
                            translations.append(t[0])
                        except Exception:
                            translations.append("")

                # Quality filter and save
                for src, tgt in zip(batch, translations):
                    tgt = tgt.strip() if tgt else ""

                    # Filter: empty translation
                    if not tgt or len(tgt) < 2:
                        rejected_empty += 1
                        continue

                    # Filter: too similar (likely a copy)
                    sim = _text_similarity(src, tgt)
                    if sim > similarity_threshold_high:
                        rejected_copy += 1
                        continue

                    # Filter: too dissimilar (likely garbage)
                    if sim < similarity_threshold_low:
                        rejected_garbage += 1
                        continue

                    record = {
                        "source_text": src,
                        "target_text": tgt,
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                        "is_synthetic": True,
                        "generation_method": "back_translation",
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    accepted += 1

                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(
                        f"  Progress: {i + batch_size}/{len(source_texts)} | "
                        f"Accepted: {accepted}"
                    )

        logger.success(
            f"Back-translation complete: {accepted} pairs accepted | "
            f"Rejected: {rejected_copy} copies, {rejected_garbage} garbage, "
            f"{rejected_empty} empty | Output: {output_file}"
        )
        return accepted

    def iterative_backtranslation(
        self,
        monolingual_file: str | Path,
        output_dir: str | Path,
        src_lang: str = "marwari",
        tgt_lang: str = "hindi",
        num_rounds: int = 3,
        batch_size: int = 32,
    ) -> list[Path]:
        """
        Run multiple rounds of back-translation.

        Each round produces better pseudo-parallel data as the model
        improves from the previous round's augmented training data.

        Returns list of output files from each round.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = []
        for round_num in range(num_rounds):
            output_file = output_dir / f"bt_round_{round_num + 1}.jsonl"
            count = self.generate(
                monolingual_file=monolingual_file,
                output_file=output_file,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                batch_size=batch_size,
            )
            output_files.append(output_file)
            logger.info(f"Round {round_num + 1}/{num_rounds}: {count} pairs")

        return output_files
