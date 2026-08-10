"""
Local MPS Fine-Tuning & Model Training Script for Rajasthani Translation

Runs directly on Apple Silicon GPU (MPS) to train/fine-tune the translation model
on local Rajasthani parallel data.

Scope warning: the only genuine parallel data in this repository is the 12 curated
entries under data/linguistic/, which expand to 24 bidirectional pairs. That is far
too little to produce a usable dialect checkpoint. This script is a working training
harness, not a route to a publishable model.
"""

import os
import json
from pathlib import Path
from loguru import logger
import torch
from src.mt.model import IndicTrans2MT
from src.mt.trainer import MTTrainer

# Below this, a fine-tune memorises the training set rather than learning the mapping.
MIN_TRAINABLE_PAIRS = 200


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

    # 2. Karya data is deliberately NOT added here.
    #
    # This block previously emitted 500 rows with source_text == target_text, labelled
    # "self-reconstruction for dialect vocabulary alignment". Two problems made it harmful
    # rather than merely useless:
    #   - The cached Karya text is standard Hindi, not dialect text, so the pairs taught
    #     Hindi -> Hindi copying under a "rajasthani -> hindi" label.
    #   - Identity pairs train the decoder toward the copy function. On a seq2seq MT model
    #     that degrades translation instead of teaching vocabulary.
    # Karya is a speech corpus; its value is ASR acoustics once the audio is fetched.
    # Re-adding it to MT requires genuine dialect<->Hindi pairs, not self-mapped rows.

    # Guard: identity pairs train the decoder toward copying its input, which is the
    # opposite of what a translation fine-tune needs. Drop them whatever the source.
    identity = [r for r in records if r["source_text"].strip() == r["target_text"].strip()]
    if identity:
        records = [r for r in records if r["source_text"].strip() != r["target_text"].strip()]
        logger.warning(f"Dropped {len(identity)} identity pairs (source_text == target_text).")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Prepared combined dataset with {len(records)} pairs at {output_path}")

    if len(records) < MIN_TRAINABLE_PAIRS:
        logger.warning(
            f"Only {len(records)} pairs available (minimum useful size ~{MIN_TRAINABLE_PAIRS}). "
            "This is the 12 curated linguistic entries expanded bidirectionally, nothing more. "
            "Fine-tuning on this will overfit immediately and the checkpoint will not be a "
            "meaningful dialect model. Fetch real parallel data before citing any result."
        )

    return output_path

def main():
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    logger.info(f"Starting Local Translation Training on Device: {device.upper()}")

    dataset_path = prepare_combined_mt_dataset(Path("data/processed/mt_mps_train.jsonl"))

    # Initialize model on MPS
    mt_model = IndicTrans2MT(device=device)
    trainer = MTTrainer(mt_model, device=device)

    # Launch local training with small batch size for low memory footprint
    logger.info("Executing training loop on Apple Silicon MPS (batch_size=1, grad_accum=8)...")
    output_checkpoint = trainer.train_manual(
        dialect_data=dataset_path,
        output_dir="models/checkpoints/mt_mps",
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=8
    )

    logger.success(f"MPS Local Training Finished! Saved checkpoint to {output_checkpoint}")

if __name__ == "__main__":
    main()
