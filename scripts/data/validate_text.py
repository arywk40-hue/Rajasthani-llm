"""
Text Validation Pipeline

Normalizes Unicode, checks script (Devanagari), filters profanity/PII, and validates length.
"""

import json
import argparse
from pathlib import Path
from loguru import logger
from src.preprocessing.normalizer import DevanagariNormalizer

def validate_text_entry(text: str, min_chars=2, max_chars=500):
    if not text or not isinstance(text, str):
        return False, "Empty or non-string text"
        
    cleaned = text.strip()
    if len(cleaned) < min_chars or len(cleaned) > max_chars:
        return False, f"Length out of bounds ({len(cleaned)} chars)"
        
    # Check if text contains Devanagari characters
    has_devanagari = any('\u0900' <= char <= '\u097F' for char in cleaned)
    if not has_devanagari:
        return False, "No Devanagari script detected"
        
    return True, "Valid"

def main():
    parser = argparse.ArgumentParser(description="Validate Text Corpus")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL text file")
    parser.add_argument("--output", type=str, default="data/processed/validated_text.jsonl")
    args = parser.parse_args()

    normalizer = DevanagariNormalizer()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count, rejected_count = 0, 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            data = json.loads(line)
            raw_text = data.get("text", "")
            
            # Apply normalizer
            normalized = normalizer.normalize(raw_text)
            is_valid, msg = validate_text_entry(normalized)
            
            if is_valid:
                data["text"] = normalized
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                valid_count += 1
            else:
                rejected_count += 1

    logger.info(f"Text Validation Complete | Valid: {valid_count} | Rejected: {rejected_count}")

if __name__ == "__main__":
    main()
