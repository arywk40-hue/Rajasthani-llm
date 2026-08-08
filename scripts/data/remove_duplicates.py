"""
Duplicate Removal Script

Removes duplicate text and audio utterances using hash matching.
"""

import json
import argparse
from pathlib import Path
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Remove duplicates from dataset")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    seen_hashes = set()
    unique_count, duplicate_count = 0, 0
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            record = json.loads(line)
            content = record.get("text", "") or record.get("audio", "")
            h = hash(content)
            
            if h in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(h)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                unique_count += 1

    logger.info(f"Deduplication Complete | Unique: {unique_count} | Duplicates Removed: {duplicate_count}")

if __name__ == "__main__":
    main()
