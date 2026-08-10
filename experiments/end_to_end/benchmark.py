"""
End-to-End S2ST & Scenario Benchmark — PLACEHOLDER, NOT A MEASUREMENT

!!! THIS SCRIPT DOES NOT INVOKE THE PIPELINE !!!

The loop below writes the string literals "PASS", 850 and "95%" for four use-case rows.
No ASR, MT or TTS component is called; nothing is timed; no success rate is counted; no
domain test sets exist in this repository.

Outputs to results/end_to_end_results.PLACEHOLDER.csv (filename intentionally marked).

To make this a real benchmark:
1. Build a per-domain test set (agriculture, healthcare, government, education) of
   dialect audio with reference transcripts and reference translations.
2. Run audio -> WhisperASR.transcribe -> IndicTrans2MT.translate -> TTS synthesis.
3. Time each stage, and derive the success rate from a defined pass criterion
   (e.g. CER below a threshold plus a non-empty translation and synthesis).
Blocked on the same missing audio as the ASR benchmark.
"""

import csv
import argparse
from pathlib import Path
from loguru import logger

USE_CASES = ["Agriculture", "Healthcare", "Government", "Education"]

BANNER = (
    "PLACEHOLDER BENCHMARK — the ASR/MT/TTS pipeline is never invoked. PASS, 850ms and "
    "95% are hardcoded literals, not results. Do not cite them as deployment validation."
)


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Scenario Benchmark")
    parser.add_argument("--output_csv", type=str, default="results/end_to_end_results.PLACEHOLDER.csv")
    args = parser.parse_args()

    logger.warning(BANNER)

    results = []
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for uc in USE_CASES:
        # Every value below is a literal. Nothing is executed or timed.
        results.append({
            "use_case": uc,
            "asr_status": "NOT_RUN",
            "mt_status": "NOT_RUN",
            "tts_status": "NOT_RUN",
            "e2e_status": "NOT_RUN",
            "avg_latency_ms": "",
            "success_rate": "",
            "measured": "false",
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "use_case", "asr_status", "mt_status", "tts_status",
            "e2e_status", "avg_latency_ms", "success_rate", "measured",
        ])
        writer.writeheader()
        writer.writerows(results)

    logger.warning(f"Placeholder output written to {out_csv}")
    logger.warning(BANNER)

if __name__ == "__main__":
    main()
