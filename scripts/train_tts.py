"""
TTS Fine-tuning Script — AI4Bharat Indic-TTS

Fine-tunes the AI4Bharat Indic-TTS Rajasthani model on per-dialect speaker data.
Uses NeMo or HuggingFace backends depending on availability.

Strategy from hackathon plan:
- Track 3: Fine-tune FastPitch + HiFi-GAN on 1-2 hours per dialect
- Even small amounts of dialect data adapt prosody significantly

Usage:
    # Fine-tune on Marwari speaker data
    python scripts/train_tts.py --dialect marwari --speaker-data data/raw/marwari_speaker.jsonl

    # Using HuggingFace SpeechT5 backend
    python scripts/train_tts.py --backend hf --dialect mewari --speaker-data ...
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tts.fastpitch import IndicTTS
from src.tts.trainer import TTSTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Indic-TTS on dialect data")
    parser.add_argument("--dialect", type=str, required=True, 
                        choices=["marwari", "mewari", "dhundhari", "hadoti", "mewati", "bagri"])
    parser.add_argument("--speaker-data", type=str, required=True, 
                        help="JSONL with audio_path, text, speaker_id for target dialect speaker")
    parser.add_argument("--output-dir", type=str, default="checkpoints/tts_dialect")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "nemo", "hf"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting TTS Fine-tuning for {args.dialect}")

    # Initialize IndicTTS (loads base Rajasthani model)
    tts = IndicTTS(device=None if args.device == "auto" else args.device)
    
    # Verify we have data
    speaker_data = Path(args.speaker_data)
    if not speaker_data.exists():
        logger.error(f"Speaker data not found: {speaker_data}")
        sys.exit(1)

    logger.info("Base model loaded. Starting fine-tuning...")
    logger.warning("TTS fine-tuning requires NeMo or HF SpeechT5 training scripts.")
    logger.info("For NeMo: Use nemo/scripts/tts/fastpitch_finetune.py with --train-manifest")
    logger.info("For HF: Use custom training loop with SpeechT5ForTextToSpeech")
    
    # TODO: Implement actual fine-tuning loop
    # This is a placeholder - real implementation needs:
    # 1. DataLoader for audio+text pairs
    # 2. FastPitch loss (mel + pitch + duration)
    # 3. HiFi-GAN loss (adversarial + feature matching + mel)
    # 4. Checkpointing and evaluation (MOS/PB-intelligibility)

    logger.success("TTS Training Script - Placeholder Complete")
    logger.info("Next: Implement NeMo/HF fine-tuning loop in src/tts/trainer.py")


if __name__ == "__main__":
    main()