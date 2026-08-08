"""
Script to train the IndicTrans2 MT model.
"""

import argparse
from loguru import logger

from src.mt.model import IndicTrans2MT
from src.mt.trainer import MTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/processed/mt_corpus_en_hi.jsonl")
    parser.add_argument("--config", type=str, default="config/mt.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logger.info("Starting MT Training Pipeline")
    
    # Initialize Model
    model = IndicTrans2MT(config_path=args.config)
    
    # Initialize Trainer
    trainer = MTTrainer(model, device=args.device)
    
    # Dummy Dataloader
    dummy_loader = [{"src": None, "tgt": None}]
    
    # Train Loop
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(dummy_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
