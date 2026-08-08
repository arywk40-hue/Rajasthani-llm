"""
Script to train the FastConformer ASR model.
"""

import argparse
from pathlib import Path
from loguru import logger

from src.asr.model import FastConformerASR
from src.asr.data_module import ASRDataModule
from src.asr.trainer import ASRTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/processed/asr_corpus.jsonl")
    parser.add_argument("--val_data", type=str, default="data/processed/asr_val.jsonl")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer_asr.model")
    parser.add_argument("--config", type=str, default="config/asr.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logger.info("Starting ASR Training Pipeline")
    
    # 1. Initialize DataModule (dummy for now if files don't exist)
    if not Path(args.train_data).exists():
        logger.warning(f"Training data not found at {args.train_data}. Skipping dataloader initialization.")
        return
        
    data_module = ASRDataModule(args.train_data, args.val_data, args.tokenizer)
    train_loader = data_module.train_dataloader()
    
    # 2. Initialize Model
    model = FastConformerASR(config_path=args.config)
    
    # 3. Initialize Trainer
    trainer = ASRTrainer(model, device=args.device)
    
    # 4. Train Loop
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")
        
    trainer.save_checkpoint("models/asr_fastconformer_final.pt")

if __name__ == "__main__":
    main()
