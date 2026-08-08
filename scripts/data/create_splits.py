"""
Train/Val/Test Split Creator

Creates reproducible 80/10/10 train/validation/test splits for datasets.
"""

import json
import random
import argparse
from pathlib import Path
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Create Train/Val/Test Splits")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_dir", type=str, default="data/processed/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    random.shuffle(records)
    total = len(records)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train_records), ("val", val_records), ("test", test_records)]:
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Splits Created | Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

if __name__ == "__main__":
    main()
