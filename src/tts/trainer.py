"""
TTS Trainer Module

Handles the two-stage training loop (Acoustic model first, Vocoder second).
"""

import torch
from loguru import logger
from src.tts.fastpitch import FastPitchAcoustic
from src.tts.hifigan import HiFiGANVocoder


class TTSTrainer:
    """Trainer for the TTS pipeline."""
    
    def __init__(self, acoustic: FastPitchAcoustic, vocoder: HiFiGANVocoder, device: str = "cpu"):
        self.acoustic = acoustic
        self.vocoder = vocoder
        self.device = torch.device(device)
        self.acoustic.to(self.device)
        self.vocoder.to(self.device)
        
        self.opt_acoustic = torch.optim.Adam(self.acoustic.parameters(), lr=2e-4)
        self.opt_vocoder = torch.optim.Adam(self.vocoder.parameters(), lr=2e-4)
        logger.info(f"TTSTrainer initialized on {self.device}")

    def train_acoustic_epoch(self, dataloader):
        """Train FastPitch."""
        self.acoustic.train()
        total_loss = 0
        for batch in dataloader:
            loss = torch.tensor(0.0, requires_grad=True) # Placeholder
            self.opt_acoustic.zero_grad()
            loss.backward()
            self.opt_acoustic.step()
            total_loss += loss.item()
        return total_loss / max(1, len(dataloader))

    def train_vocoder_epoch(self, dataloader):
        """Train HiFi-GAN."""
        self.vocoder.train()
        total_loss = 0
        for batch in dataloader:
            loss = torch.tensor(0.0, requires_grad=True) # Placeholder
            self.opt_vocoder.zero_grad()
            loss.backward()
            self.opt_vocoder.step()
            total_loss += loss.item()
        return total_loss / max(1, len(dataloader))
