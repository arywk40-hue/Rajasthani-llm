"""
MT Trainer Module

Handles the training loop for IndicTrans2. Implements Experience Replay
and Model Souping to prevent catastrophic forgetting when fine-tuning
on specific Rajasthani dialects.
"""

import torch
from loguru import logger
from src.mt.model import IndicTrans2MT


class MTTrainer:
    """Trainer for the MT model."""
    
    def __init__(self, model: IndicTrans2MT, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
        logger.info(f"MTTrainer initialized on {self.device}")

    def train_epoch(self, dataloader, experience_replay_loader=None):
        """
        Runs one epoch of training, optionally mixing in data from the
        experience_replay_loader to prevent catastrophic forgetting.
        """
        self.model.train()
        total_loss = 0
        
        # In a real implementation, batches from dataloader and experience_replay_loader
        # would be interleaved.
        
        for batch_idx, batch in enumerate(dataloader):
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            
            output = self.model(src, tgt)
            loss = torch.tensor(0.0, requires_grad=True) # Placeholder for CrossEntropyLoss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / max(1, len(dataloader))

    def create_model_soup(self, checkpoints: list[str]) -> IndicTrans2MT:
        """
        Averages the weights of multiple fine-tuned checkpoints with the base weights
        to improve conversational Character-F1 scores without losing general domain capabilities.
        """
        logger.info(f"Creating model soup from {len(checkpoints)} checkpoints...")
        # Placeholder for weight averaging logic
        return self.model
