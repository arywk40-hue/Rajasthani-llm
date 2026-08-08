"""
Script to fetch datasets from HuggingFace for the Rajasthani Dialect AI.

Usage:
    # Fetch all datasets (text metadata only — fast)
    python scripts/fetch_data.py

    # Fetch with audio files (for Whisper fine-tuning — slower, needs disk space)
    python scripts/fetch_data.py --with-audio --max-vaani 1000

    # Fetch specific dialects only
    python scripts/fetch_data.py --dialects marwari bagri
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetch_datasets import DatasetFetcher, ALL_DIALECTS


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch datasets for Rajasthani Dialect AI")
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--dialects", nargs="+", default=ALL_DIALECTS)
    parser.add_argument("--max-vaani", type=int, default=5000, help="Max samples per dialect from VAANI")
    parser.add_argument("--max-karya", type=int, default=10000, help="Max samples from Karya")
    parser.add_argument("--with-audio", action="store_true", help="Also download audio files")
    parser.add_argument("--only", choices=["vaani", "karya", "all"], default="all")
    args = parser.parse_args()

    fetcher = DatasetFetcher(output_dir=args.output_dir)

    if args.only in ("vaani", "all"):
        if args.with_audio:
            print(f"📥 Fetching VAANI with audio for: {args.dialects}")
            fetcher.fetch_vaani_with_audio(
                dialects=args.dialects,
                max_samples_per_dialect=args.max_vaani,
            )
        else:
            print(f"📥 Fetching VAANI metadata for: {args.dialects}")
            fetcher.fetch_vaani(
                dialects=args.dialects,
                max_samples_per_dialect=args.max_vaani,
            )

    if args.only in ("karya", "all"):
        print(f"📥 Fetching Karya (speech-rj-hi)...")
        fetcher.fetch_karya(max_samples=args.max_karya)

    print("\n📊 Download Report:")
    for name, count in fetcher.report().items():
        print(f"  {name}: {count} records")
    print("✅ Done!")


if __name__ == "__main__":
    main()
