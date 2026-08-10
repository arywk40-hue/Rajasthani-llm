"""
TTS Benchmarking Suite — PLACEHOLDER, NOT A MEASUREMENT

!!! THIS SCRIPT DOES NOT SYNTHESIZE OR EVALUATE ANY AUDIO !!!

MOS (Mean Opinion Score) is by definition an average of human listener ratings. No
listening study has been conducted for this project. The loop below passes the constant
4.2 into `evaluator.evaluate_tts()` and reads the same value back; `mos_pronunciation`
is the unused literal 4.1; `pb_pass` is hardcoded `hits=9, total=10`; latency is the
literal 240 and is never timed.

Outputs to results/tts_results.PLACEHOLDER.csv (filename intentionally marked).

To make this a real evaluation:
1. Produce a working TTS path (src/tts/hifigan.py is a 5-layer placeholder, not
   HiFi-GAN; the Bhashini cloud route needs a verified service ID and API key).
2. Synthesize a phonetically balanced sentence set per dialect.
3. Run a blind listening study with native speakers, recording one MOSRecord per
   (audio_id, reviewer) pair via MOSScorecard.add().
4. Time synthesis calls for real latency figures.
"""

import csv
import argparse
from pathlib import Path
from loguru import logger
from src.evaluation.metrics import Evaluator

DIALECTS = ["marwari", "mewari", "dhundhari", "hadoti", "mewati", "bagri"]

BANNER = (
    "PLACEHOLDER BENCHMARK — no audio is synthesized and no human rated anything. "
    "MOS requires a listening study; these are hardcoded constants. Do not cite them "
    "as naturalness or intelligibility scores."
)


def main():
    parser = argparse.ArgumentParser(description="Run TTS Benchmark across Rajasthani Dialects")
    parser.add_argument("--output_csv", type=str, default="results/tts_results.PLACEHOLDER.csv")
    args = parser.parse_args()

    logger.warning(BANNER)

    evaluator = Evaluator()
    results = []

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for dialect in DIALECTS:
        # Passing 4.2 in and reading 4.2 back out is not a measurement.
        evaluator.evaluate_tts(
            audio_id=f"sample_{dialect}",
            score=4.2,
            reviewer="PLACEHOLDER-no-human-reviewer",
            dialect=dialect,
        )
        pb_pass = evaluator.evaluate_tts_pb(hits=9, total=10)

        results.append({
            "model": "none-NO-SYNTHESIS-PERFORMED",
            "dialect": dialect,
            "mos_naturalness": 4.2,
            "mos_pronunciation": 4.1,
            "pb_pass": "PASS" if pb_pass else "FAIL",
            "latency_ms": 240,  # hardcoded constant, never timed
            "measured": "false",
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "dialect", "mos_naturalness", "mos_pronunciation",
            "pb_pass", "latency_ms", "measured",
        ])
        writer.writeheader()
        writer.writerows(results)

    logger.warning(f"Placeholder output written to {out_csv}")
    logger.warning(BANNER)

if __name__ == "__main__":
    main()
