"""
Build Corpus CLI — Month 1-2 roadmap exit criteria.

Runs the full data curation pipeline:
    1. Ingest raw datasets (VAANI, BPCC, LDC-IL, IndicTTS, Karya)
    2. Apply non-negotiable NFC / Nukta / ळ normalization + quality filtering
    3. Split into deterministic train/val splits
    4. Emit the LDC-IL GOLDEN evaluation set (held out of training)
    5. Train SentencePiece BPE tokenizers on the normalized text

Deliverable: "Clean, unified corpus + tokenizer ready" (implementation.md §5, Months 1-2).

Usage:
    python scripts/build_corpus.py --raw-dir data/raw --out-dir data/processed
    python scripts/build_corpus.py --corpora mt --src en --tgt hi   # subset
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.data.corpus_builder import CorpusBuilder

ALL_DIALECTS = ["marwari", "mewari", "dhundhari", "hadoti", "mewati", "bagri"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Rajasthani dialect data curation pipeline")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Root dir containing raw datasets")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="Output dir for processed corpora")
    parser.add_argument("--config", type=str, default="config/base.yaml", help="Normalizer config path")
    parser.add_argument(
        "--corpora",
        type=str,
        nargs="+",
        choices=["asr", "mt", "tts", "golden", "tokenizer"],
        default=["asr", "mt", "tts", "golden", "tokenizer"],
        help="Which corpora/steps to build (default: all)",
    )
    parser.add_argument("--src", type=str, default="en", help="MT source language (e.g. en, hi)")
    parser.add_argument("--tgt", type=str, default="hi", help="MT target language (e.g. hi, en)")
    parser.add_argument("--dialects", type=str, nargs="+", default=ALL_DIALECTS, help="Dialects for the ASR corpus")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split fraction (0..1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the deterministic split")
    parser.add_argument("--tokenizer-prefix", type=str, default="models/tokenizer", help="BPE model prefix")
    parser.add_argument("--tokenizer-config", type=str, default="config/tokenizer.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = CorpusBuilder(
        raw_data_dir=args.raw_dir,
        output_dir=args.out_dir,
        normalizer_config_path=args.config,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    corpora = set(args.corpora)

    if "asr" in corpora:
        train_path, val_path = builder.build_asr_corpus(args.dialects)
        logger.success(f"ASR corpus -> {train_path} / {val_path}")

    if "mt" in corpora:
        train_path, val_path = builder.build_mt_corpus(args.src, args.tgt)
        logger.success(f"MT corpus -> {train_path} / {val_path}")

    if "tts" in corpora:
        train_path, val_path = builder.build_tts_corpus()
        logger.success(f"TTS corpus -> {train_path} / {val_path}")

    if "golden" in corpora:
        golden_path = builder.build_golden_eval_corpus("rj", "hi")
        logger.success(f"Golden eval corpus -> {golden_path}")

    if "tokenizer" in corpora:
        _train_tokenizers(args)

    logger.success("Data curation pipeline complete.")


def _train_tokenizers(args: argparse.Namespace) -> None:
    """
    Train SentencePiece BPE tokenizers on the freshly built corpora.
    Requires processed corpus JSONL files to exist.
    """
    from src.tokenizer.bpe_trainer import BPETrainer

    trainer = BPETrainer(config_path=args.tokenizer_config)

    jobs: list[tuple[Path, str]] = []

    asr_train = Path(args.out_dir) / "asr_corpus_train.jsonl"
    if asr_train.exists():
        jobs.append((asr_train, f"{args.tokenizer_prefix}_asr"))

    mt_train = Path(args.out_dir) / f"mt_corpus_{args.src}_{args.tgt}_train.jsonl"
    if mt_train.exists():
        jobs.append((mt_train, f"{args.tokenizer_prefix}_mt"))

    tts_train = Path(args.out_dir) / "tts_corpus_train.jsonl"
    if tts_train.exists():
        jobs.append((tts_train, f"{args.tokenizer_prefix}_tts"))

    if not jobs:
        logger.warning("No processed corpora found; skipping tokenizer training.")
        return

    for corpus_path, prefix in jobs:
        temp_text = f"{prefix}_temp_input.txt"
        try:
            trainer.prepare_text_file(corpus_path, temp_text)
            trainer.train(temp_text, prefix)
        finally:
            Path(temp_text).unlink(missing_ok=True)

    logger.success(f"Trained {len(jobs)} tokenizers under '{Path(args.tokenizer_prefix).parent}'")


if __name__ == "__main__":
    sys.exit(main())