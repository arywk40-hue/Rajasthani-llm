"""
Script to train the two-stage TTS pipeline.
"""

import argparse
from loguru import logger

from src.tts.fastpitch import FastPitchAcoustic
from src.tts.hifigan import HiFiGANVocoder
from src.tts.trainer import TTSTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/processed/tts_corpus.jsonl")
    parser.add_argument("--config", type=str, default="config/tts.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logger.info("Starting TTS Training Pipeline")
    
    acoustic = FastPitchAcoustic(config_path=args.config)
    vocoder = HiFiGANVocoder(config_path=args.config)
    trainer = TTSTrainer(acoustic, vocoder, device=args.device)
    
    dummy_loader = []
    
    # Train Acoustic
    for epoch in range(args.epochs // 2):
        loss = trainer.train_acoustic_epoch(dummy_loader)
        logger.info(f"Acoustic Epoch {epoch+1} - Loss: {loss:.4f}")
        
    # Train Vocoder
    for epoch in range(args.epochs // 2):
        loss = trainer.train_vocoder_epoch(dummy_loader)
        logger.info(f"Vocoder Epoch {epoch+1} - Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
