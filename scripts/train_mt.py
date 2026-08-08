"""
Script to train the IndicTrans2 MT model.

Supports two modes:
1. HuggingFace Trainer (recommended for multi-GPU, mixed precision)
2. Manual Trainer (for full control over experience replay scheduling)

Usage:
    # Basic fine-tuning
    python scripts/train_mt.py --train-data data/processed/mt_dialect_train.jsonl

    # Fine-tuning with experience replay (prevent catastrophic forgetting)
    python scripts/train_mt.py \
        --train-data data/processed/mt_dialect_train.jsonl \
        --general-data data/processed/mt_bpcc_train.jsonl \
        --replay-ratio 0.15 \
        --eval-data data/processed/ldcil_golden.jsonl
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mt.model import IndicTrans2MT
from src.mt.trainer import MTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train IndicTrans2 MT model")
    parser.add_argument("--train-data", type=str, required=True, help="Path to dialect training JSONL")
    parser.add_argument("--eval-data", type=str, default=None, help="Path to evaluation JSONL (golden set)")
    parser.add_argument("--general-data", type=str, default=None, help="Path to general BPCC data for experience replay")
    
    parser.add_argument("--model-name", type=str, default="indic-indic-dist-200M", help="HuggingFace model variant")
    parser.add_argument("--config", type=str, default="config/mt.yaml", help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="checkpoints/mt_indictrans2")
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--replay-ratio", type=float, default=0.15, help="Ratio of general data to mix in")
    
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--manual-loop", action="store_true", help="Use manual training loop instead of HF Trainer")
    
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting IndicTrans2 MT Training Pipeline")
    
    # Initialize Model
    model = IndicTrans2MT(
        config_path=args.config,
        model_name=args.model_name,
        device=None if args.device == "auto" else args.device,
    )
    
    # Initialize Trainer
    trainer = MTTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
    )
    
    # Train
    if args.manual_loop:
        logger.info("Using Manual Training Loop")
        trainer.train_manual(
            dialect_data=args.train_data,
            general_data=args.general_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            replay_ratio=args.replay_ratio,
            eval_data=args.eval_data,
        )
    else:
        logger.info("Using HuggingFace Trainer")
        trainer.train_with_hf_trainer(
            train_data=args.train_data,
            eval_data=args.eval_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            general_data=args.general_data,
            replay_ratio=args.replay_ratio,
        )

    logger.success("MT Training Script Completed")


if __name__ == "__main__":
    main()
