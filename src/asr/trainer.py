"""
ASR Trainer Module

Handles the training loop for the FastConformer ASR model, including loss
calculation (hybrid CTC/TDT) and optimizer steps.
"""

import torch
from loguru import logger
from src.asr.model import FastConformerASR


class ASRTrainer:
    """Trainer for the ASR model."""
    
    def __init__(self, model: FastConformerASR, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        logger.info(f"ASRTrainer initialized on {self.device}")

    def train_epoch(self, dataloader):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Skeleton logic for forward pass and loss
            waveforms = batch["waveforms"]
            # Example: extract features and forward pass
            # features = extract_features(waveforms)
            # tdt_logits, ctc_logits = self.model(features, lengths)
            
            loss = torch.tensor(0.0, requires_grad=True) # Placeholder
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / max(1, len(dataloader))

    def save_checkpoint(self, path: str):
        """Saves model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
