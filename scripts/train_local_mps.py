"""
Local MPS Fine-Tuning & Model Training Script for Rajasthani Translation

Runs directly on Apple Silicon GPU (MPS) to train/fine-tune the translation model
on local Rajasthani parallel data.
"""

import os
import json
from pathlib import Path
from loguru import logger
import torch
from src.mt.model import IndicTrans2MT
from src.mt.trainer import MTTrainer

def prepare_combined_mt_dataset(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    # 1. Add linguistic idioms & expressions
    linguistic_dir = Path("data/linguistic")
    if linguistic_dir.exists():
        for json_file in linguistic_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    source = item.get("source", "")
                    target = item.get("reference_translation", "") or item.get("literal_translation", "")
                    dialect = item.get("dialect", "marwari").lower()
                    if source and target:
                        records.append({
                            "source_text": source,
                            "target_text": target,
                            "src_lang": dialect,
                            "tgt_lang": "hindi"
                        })
                        records.append({
                            "source_text": target,
                            "target_text": source,
                            "src_lang": "hindi",
                            "tgt_lang": dialect
                        })

    # 2. Add Karya data
    karya_path = Path("data/raw/karya/karya_rajasthan.jsonl")
    if karya_path.exists():
        with open(karya_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 500: break # Use top 500 lines for fast local MPS training
                if not line.strip(): continue
                rec = json.loads(line)
                text = rec.get("text", "")
                if text:
                    records.append({
                        "source_text": text,
                        "target_text": text, # Self-reconstruction for dialect vocabulary alignment
                        "src_lang": rec.get("dialect", "rajasthani"),
                        "tgt_lang": "hindi"
                    })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Prepared combined dataset with {len(records)} pairs at {output_path}")
    return output_path

def main():
    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    logger.info(f"Starting Local Translation Training on Device: {device.upper()}")

    dataset_path = prepare_combined_mt_dataset(Path("data/processed/mt_mps_train.jsonl"))

    # Initialize model on MPS
    mt_model = IndicTrans2MT(device=device)
    trainer = MTTrainer(mt_model, device=device)

    # Launch local training
    logger.info("Executing training loop on Apple Silicon MPS...")
    output_checkpoint = trainer.train_manual(
        dialect_data=dataset_path,
        output_dir="models/checkpoints/mt_mps",
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=2
    )

    logger.success(f"MPS Local Training Finished! Saved checkpoint to {output_checkpoint}")

if __name__ == "__main__":
    main()
