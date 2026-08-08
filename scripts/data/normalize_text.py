"""
Standalone Text Normalization Utility

Batch-normalizes text files using DevanagariNormalizer.
"""

import json
import argparse
from pathlib import Path
from loguru import logger
from src.preprocessing.normalizer import DevanagariNormalizer

def main():
    parser = argparse.ArgumentParser(description="Normalize Devanagari Text Corpus")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    normalizer = DevanagariNormalizer()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            record = json.loads(line)
            if "text" in record:
                record["text"] = normalizer.normalize(record["text"])
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Normalized {count} entries into {out_path}")

if __name__ == "__main__":
    main()
