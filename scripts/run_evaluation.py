"""
Run the evaluation harness against the LDC-IL GOLDEN test set (roadmap §6).

The LDC-IL Rajasthani corpus (31,096 words / 5,332 sentences) is held strictly
out of training to avoid leakage. This script:
    - loads the golden parallel corpus
    - runs a translation model (or a mock) over the source side
    - computes chrF++ (and optionally CER/WER for ASR transcripts)
    - emits a JSONL scorecard + prints a human-readable summary

Usage:
    python scripts/run_evaluation.py --golden data/processed/ldcil_golden.jsonl
    python scripts/run_evaluation.py --golden ... --mock-substitution   # no-model sanity run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from loguru import logger

from src.evaluation.metrics import COMETWrapper, Evaluator


def load_golden(path: str | Path) -> list[tuple[str, str]]:
    """Load (source, reference) pairs from a JSONL golden corpus file."""
    path = Path(path)
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            src = record.get("source_text") or record.get("source")
            ref = record.get("target_text") or record.get("reference") or record.get("text")
            if src and ref:
                pairs.append((src, ref))
    return pairs


def default_translator(text: str) -> str:
    """Identity mock: substitutes for a live translation model in CI runs."""
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the LDC-IL golden evaluation set")
    parser.add_argument("--golden", type=str, default="data/processed/ldcil_golden.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Only score first N pairs")
    parser.add_argument("--output", type=str, default="reports/mt_chrf.jsonl")
    parser.add_argument(
        "--mock-substitution",
        action="store_true",
        help="Use the identity mock translator instead of a real model (sanity check).",
    )
    parser.add_argument("--translator-module", type=str, default=None,
                        help="Import path 'module:function' returning a text->text callable.")
    args = parser.parse_args()

    logger.info("Running Evaluation Pipeline")
    pairs = load_golden(args.golden)
    if args.limit:
        pairs = pairs[: args.limit]
    logger.info(f"Loaded {len(pairs)} golden pairs from {args.golden}")

    # Resolve the translator callable
    translator: Callable[[str], str]
    if args.translator_module:
        module_path, _, func_name = args.translator_module.partition(":")
        import importlib

        translator = getattr(importlib.import_module(module_path), func_name)
        logger.info(f"Using translator from {args.translator_module}")
    elif args.mock_substitution:
        translator = default_translator
        logger.info("Using MOCK substitution translator (identity) — sanity run only.")
    else:
        logger.error(
            "No translator configured. Pass --translator-module 'pkg.module:func' "
            "or --mock-substitution for a mock run."
        )
        sys.exit(1)

    evaluator = Evaluator()
    hypotheses = [translator(src) for src, _ in pairs]
    references = [ref for _, ref in pairs]

    report = evaluator.evaluate_mt(hypotheses, references)
    if args.output:
        evaluator.save_report(report, args.output)

    logger.info(report.summary())
    print("\n=== MT Evaluation (chrF++) ===")
    print(f"  Pairs evaluated : {report.samples}")
    print(f"  Mean chrF++     : {report.mean_chrf:.4f}")
    print()

    # Optional COMET (only if the optional dependency is installed)
    comet = evaluator.get_comet()
    if comet.available:
        logger.info("COMET available; scoring is left to a dedicated script.")
    else:
        logger.info("COMET not installed; using chrF++ as the primary MT metric (roadmap allows).")

    if not args.limit and not args.output:
        logger.warning("No output path specified; report not persisted.")


if __name__ == "__main__":
    sys.exit(main())