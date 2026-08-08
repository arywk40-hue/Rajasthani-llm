"""
Script to train the Whisper ASR model.

Implements the 3-track hackathon strategy:
Track 1: Benchmark zero-shot on test data
Track 2: Fine-tune on dialect data

Usage:
    # Track 1: Zero-shot benchmark
    python scripts/train_asr.py --benchmark-only --test-data data/processed/asr_val.jsonl

    # Track 2: Fine-tune
    python scripts/train_asr.py \
        --train-data data/processed/asr_train.jsonl \
        --val-data data/processed/asr_val.jsonl \
        --epochs 10 \
        --batch-size 8
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.trainer import ASRTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper ASR model")
    
    # Modes
    parser.add_argument("--benchmark-only", action="store_true", help="Run zero-shot benchmark and exit")
    
    # Data
    parser.add_argument("--train-data", type=str, default="data/processed/asr_train.jsonl")
    parser.add_argument("--val-data", type=str, default="data/processed/asr_val.jsonl")
    parser.add_argument("--test-data", type=str, default="data/processed/asr_val.jsonl", help="Data for benchmark")
    
    # Model
    parser.add_argument("--model-name", type=str, default="indicwhisper-hindi-large-v2")
    parser.add_argument("--output-dir", type=str, default="checkpoints/asr_whisper")
    
    # Training Params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    
    # Debug limits
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting ASR Pipeline (Whisper)")
    
    # Initialize Trainer
    trainer = ASRTrainer(model_name=args.model_name)
    
    if args.benchmark_only:
        logger.info("Running Zero-Shot Benchmark (Track 1)")
        report = trainer.benchmark(
            test_data=args.test_data,
            max_samples=args.max_eval_samples
        )
        logger.success(f"Benchmark Results: CER={report.get('mean_cer', 0.0):.4f}, WER={report.get('mean_wer', 0.0):.4f}")
        return

    logger.info("Starting Fine-tuning (Track 2)")
    
    trainer.train(
        train_data=args.train_data,
        eval_data=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    
    logger.success("ASR Training Script Completed")


if __name__ == "__main__":
    main()
