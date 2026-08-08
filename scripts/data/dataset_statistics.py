"""
Dataset Statistics Summary Generator

Generates quantitative reports on audio duration, text length, and speaker counts.
"""

import json
import argparse
from pathlib import Path
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Dataset Statistics")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    args = parser.parse_args()

    total_samples = 0
    total_duration = 0.0
    speakers = set()
    dialects = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            total_samples += 1
            total_duration += rec.get("duration", 0.0)
            if "speaker_id" in rec:
                speakers.add(rec["speaker_id"])
            dialect = rec.get("dialect", "unknown")
            dialects[dialect] = dialects.get(dialect, 0) + 1

    report = f"""
Dataset Statistics Report
-------------------------
Total Samples : {total_samples}
Total Audio   : {total_duration / 3600.0:.2f} hours ({total_duration:.1f} sec)
Total Speakers: {len(speakers)}
Dialect Distribution:
"""
    for d, count in dialects.items():
        report += f"  - {d}: {count} samples\n"

    print(report)
    logger.info(f"Calculated statistics for {total_samples} records.")

if __name__ == "__main__":
    main()
