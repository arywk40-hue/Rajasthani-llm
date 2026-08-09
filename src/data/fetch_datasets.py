"""
Dataset Fetcher — Downloads and filters real datasets from HuggingFace.

Handles:
1. VAANI (ARTPARK-IISc) — spontaneous speech, filter by Rajasthan districts
2. Speech-rj-hi (Karya) — read speech from Rajasthan
3. IndicVoices-R — multi-speaker TTS data
4. BPCC (IndicTrans2) — parallel text for MT

All data is streamed via HuggingFace `datasets` to avoid downloading
the full 31,255-hour corpus when we only need Rajasthan subsets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger


# ─── Rajasthan District → Dialect Mapping ─────────────────────────────────────
# Maps Rajasthan districts to their primary dialect.
# Source: linguistic documentation in detailed-report.md

DISTRICT_DIALECT_MAP = {
    # Marwari belt (Western Rajasthan)
    "jodhpur": "marwari",
    "jaisalmer": "marwari",
    "barmer": "marwari",
    "pali": "marwari",
    "jalore": "marwari",
    "sirohi": "marwari",
    "nagaur": "marwari",
    "bikaner": "marwari",
    "churu": "marwari",
    # Mewari belt (Southern Rajasthan)
    "udaipur": "mewari",
    "rajsamand": "mewari",
    "bhilwara": "mewari",
    "chittorgarh": "mewari",
    "pratapgarh": "mewari",
    # Dhundhari belt (Eastern Rajasthan)
    "jaipur": "dhundhari",
    "dausa": "dhundhari",
    "tonk": "dhundhari",
    "sawai madhopur": "dhundhari",
    # Hadoti belt (Southeastern Rajasthan)
    "kota": "hadoti",
    "bundi": "hadoti",
    "baran": "hadoti",
    "jhalawar": "hadoti",
    # Mewati belt (Northeastern Rajasthan)
    "alwar": "mewati",
    "bharatpur": "mewati",
    "dholpur": "mewati",
    "karauli": "mewati",
    # Bagri belt (Northern Rajasthan, bordering Haryana/Punjab)
    "hanumangarh": "bagri",
    "sri ganganagar": "bagri",
    "ganganagar": "bagri",
}

# All Rajasthan districts (for broad filtering)
RAJASTHAN_DISTRICTS = set(DISTRICT_DIALECT_MAP.keys())

ALL_DIALECTS = ["marwari", "mewari", "dhundhari", "hadoti", "mewati", "bagri"]


def get_dialect_for_district(district: str) -> str:
    """Map a district name to its primary dialect."""
    return DISTRICT_DIALECT_MAP.get(district.lower().strip(), "unknown")


class DatasetFetcher:
    """
    Fetches and filters real datasets from HuggingFace for the
    Rajasthani Dialect AI project.

    Usage:
        fetcher = DatasetFetcher(output_dir="data/raw")

        # Download VAANI Rajasthan subset
        fetcher.fetch_vaani(dialects=["marwari", "bagri"])

        # Download Karya read-speech
        fetcher.fetch_karya()

        # Download BPCC Hindi-English for MT
        fetcher.fetch_bpcc_sample(src="hi", tgt="en", max_pairs=100000)
    """

    def __init__(self, output_dir: str | Path = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DatasetFetcher initialized. Output: {self.output_dir}")

    # ─── VAANI Dataset ────────────────────────────────────────────────────────

    def fetch_vaani(
        self,
        dialects: Optional[list[str]] = None,
        max_samples_per_dialect: int = 5000,
        split: str = "train",
    ) -> Path:
        """
        Fetch VAANI dataset from HuggingFace, filtered to Rajasthan districts.

        The full VAANI dataset is ~31,255 hours. We stream it and only
        download samples from Rajasthan districts, tagged by dialect.

        Args:
            dialects: Which dialects to fetch (default: all 6)
            max_samples_per_dialect: Cap per dialect to avoid imbalance
            split: HuggingFace split to use

        Returns:
            Path to output JSONL file
        """
        try:
            from datasets import load_dataset
            import datasets
        except ImportError:
            logger.error("Install `datasets`: pip install datasets")
            raise

        dialects = dialects or ALL_DIALECTS
        target_districts = {
            d: dial for d, dial in DISTRICT_DIALECT_MAP.items()
            if dial in dialects
        }

        output_path = self.output_dir / "vaani" / "vaani_rajasthan.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Fetching VAANI dataset | Dialects: {dialects} | "
            f"Districts: {len(target_districts)} | Max/dialect: {max_samples_per_dialect}"
        )

        dialect_counts: dict[str, int] = {d: 0 for d in dialects}
        total = 0

        # Define geocentric configurations for Rajasthan districts
        dialect_to_configs = {
            "marwari": ["Rajasthan_Barmer", "Rajasthan_Bikaner", "Rajasthan_Churu", "Rajasthan_Jaisalmer", "Rajasthan_Nagaur"],
            "dhundhari": ["Rajasthan_Jaipur"]
        }

        configs_to_load = []
        for dialect in dialects:
            if dialect in dialect_to_configs:
                configs_to_load.extend(dialect_to_configs[dialect])

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for config_name in configs_to_load:
                    logger.info(f"Streaming VAANI configuration: {config_name}")
                    try:
                        dataset = load_dataset(
                            "ARTPARK-IISc/Vaani",
                            config_name,
                            split=split,
                            streaming=True,
                        )
                        # Avoid audio decoding during metadata fetch
                        dataset = dataset.cast_column("audio", datasets.Audio(decode=False))
                    except Exception as e:
                        logger.error(f"Failed to load VAANI configuration {config_name}: {e}")
                        continue

                    for sample in dataset:
                        district = str(sample.get("district", sample.get("districtName", ""))).lower().strip()
                        language = str(sample.get("language", "")).lower().strip()

                        dialect = None
                        if district in target_districts:
                            dialect = target_districts[district]
                        elif language in dialects:
                            dialect = language
                        elif "rajasthani" in language or "marwari" in language:
                            dialect = "marwari"

                        if dialect is None:
                            continue

                        if dialect_counts.get(dialect, 0) >= max_samples_per_dialect:
                            continue

                        # Extract audio and transcript (correct key is transcript!)
                        text = str(sample.get("transcript", sample.get("text", sample.get("sentence", "")))).strip()
                        audio = sample.get("audio", {})

                        if not text or text == "None":
                            continue

                        record = {
                            "text": text,
                            "dialect": dialect,
                            "district": district,
                            "language": language,
                            "source": "vaani",
                            "sample_rate": audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000,
                        }

                        if isinstance(audio, dict) and "path" in audio:
                            record["audio_path"] = audio["path"]

                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
                        total += 1

                        if total % 500 == 0:
                            logger.info(f"  Fetched {total} samples so far: {dialect_counts}")

        except Exception as e:
            logger.error(f"Error fetching VAANI: {e}")

        logger.success(
            f"VAANI fetch complete: {total} total samples | "
            f"Per-dialect: {dialect_counts} | Output: {output_path}"
        )
        return output_path

    # ─── Karya (Speech-rj-hi) ─────────────────────────────────────────────────

    def fetch_karya(
        self,
        max_samples: int = 10000,
        split: str = "train",
    ) -> Path:
        """
        Fetch the Speech-rj-hi (Karya) dataset — read speech from Rajasthan.

        426,873 clips from 98 participants (58 male, 40 female) from Soda, Rajasthan.
        Clean, clearly articulated read speech — ideal for TTS training baseline.
        """
        try:
            from datasets import load_dataset
            import datasets
        except ImportError:
            logger.error("Install `datasets`: pip install datasets")
            raise

        output_path = self.output_dir / "karya" / "karya_rajasthan.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Fetching Karya (speech-rj-hi) | Max: {max_samples}")

        count = 0
        try:
            dataset = load_dataset(
                "severo/speech-rj-hi",
                split=split,
                streaming=True,
            )
            # Avoid decoding audio data to prevent torchcodec dependency during fetch
            dataset = dataset.cast_column("audio", datasets.Audio(decode=False))

            with open(output_path, "w", encoding="utf-8") as f:
                for sample in dataset:
                    text = str(sample.get("text", sample.get("sentence", ""))).strip()
                    audio = sample.get("audio", {})

                    if not text:
                        continue

                    record = {
                        "text": text,
                        "dialect": "rajasthani",  # Broad — Karya doesn't partition by dialect
                        "source": "karya",
                        "sample_rate": audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000,
                    }
                    if isinstance(audio, dict) and "path" in audio:
                        record["audio_path"] = audio["path"]

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

                    if count >= max_samples:
                        break
                    if count % 1000 == 0:
                        logger.info(f"  Karya: {count} samples fetched")

        except Exception as e:
            logger.error(f"Error fetching Karya: {e}")

        logger.success(f"Karya fetch complete: {count} samples | Output: {output_path}")
        return output_path

    # ─── VAANI Audio Download (for Whisper fine-tuning) ───────────────────────

    def fetch_vaani_with_audio(
        self,
        dialects: Optional[list[str]] = None,
        max_samples_per_dialect: int = 1000,
        output_format: str = "hf",
    ) -> Path:
        """
        Fetch VAANI data WITH audio arrays for Whisper fine-tuning.

        This version saves the actual audio arrays alongside transcripts
        in a format ready for Whisper fine-tuning (HuggingFace datasets format).

        For larger datasets, use streaming mode and save to disk incrementally.
        """
        try:
            from datasets import load_dataset, Audio, Dataset
            import soundfile as sf
        except ImportError:
            logger.error("Install: pip install datasets soundfile")
            raise

        dialects = dialects or ALL_DIALECTS
        target_districts = {
            d: dial for d, dial in DISTRICT_DIALECT_MAP.items()
            if dial in dialects
        }

        audio_dir = self.output_dir / "vaani" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.output_dir / "vaani" / "vaani_audio_metadata.jsonl"

        logger.info(
            f"Fetching VAANI with audio | Dialects: {dialects} | "
            f"Max/dialect: {max_samples_per_dialect}"
        )

        dialect_counts: dict[str, int] = {d: 0 for d in dialects}
        total = 0

        # Define geocentric configurations for Rajasthan districts
        dialect_to_configs = {
            "marwari": ["Rajasthan_Barmer", "Rajasthan_Bikaner", "Rajasthan_Churu", "Rajasthan_Jaisalmer", "Rajasthan_Nagaur"],
            "dhundhari": ["Rajasthan_Jaipur"]
        }

        configs_to_load = []
        for dialect in dialects:
            if dialect in dialect_to_configs:
                configs_to_load.extend(dialect_to_configs[dialect])

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                for config_name in configs_to_load:
                    logger.info(f"Streaming VAANI with audio configuration: {config_name}")
                    try:
                        dataset = load_dataset(
                            "ARTPARK-IISc/Vaani",
                            config_name,
                            split="train",
                            streaming=True,
                        )
                    except Exception as e:
                        logger.error(f"Failed to load VAANI configuration {config_name}: {e}")
                        continue

                    for sample in dataset:
                        district = str(sample.get("district", sample.get("districtName", ""))).lower().strip()
                        language = str(sample.get("language", "")).lower().strip()

                        dialect = target_districts.get(district)
                        if dialect is None:
                            if language in dialects:
                                dialect = language
                            else:
                                continue

                        if dialect_counts.get(dialect, 0) >= max_samples_per_dialect:
                            continue

                        # Extract audio and transcript (correct key is transcript!)
                        text = str(sample.get("transcript", sample.get("text", sample.get("sentence", "")))).strip()
                        audio = sample.get("audio", {})

                        if not text or text == "None" or not isinstance(audio, dict):
                            continue

                        # Save audio file
                        audio_array = audio.get("array")
                        sr = audio.get("sampling_rate", 16000)
                        if audio_array is not None:
                            audio_filename = f"{dialect}_{total:06d}.wav"
                            audio_path = audio_dir / audio_filename
                            try:
                                import numpy as np
                                sf.write(str(audio_path), np.array(audio_array), sr)
                            except Exception as e:
                                logger.warning(f"Could not save audio {audio_filename}: {e}")
                                continue

                            record = {
                                "audio_path": str(audio_path),
                                "text": text,
                                "dialect": dialect,
                                "district": district,
                                "sample_rate": sr,
                                "source": "vaani",
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
                            total += 1

                            if total % 100 == 0:
                                logger.info(f"  Audio saved: {total} | {dialect_counts}")

        except Exception as e:
            logger.error(f"Error fetching VAANI audio: {e}")

        logger.success(
            f"VAANI audio fetch complete: {total} samples | "
            f"Per-dialect: {dialect_counts}"
        )
        return metadata_path

    # ─── BPCC Parallel Corpus (for MT) ──────────────────────────────────────────

    def fetch_bpcc_sample(
        self,
        src: str = "hi",
        tgt: str = "en",
        max_pairs: int = 100000,
    ) -> Path:
        """
        Fetch BPCC (Bharat Parallel Corpus Collection) parallel text for MT training.

        BPCC is the training data for IndicTrans2 — 230M parallel sentence pairs
        across 22 scheduled languages. We stream and sample the required language pair.

        Args:
            src: Source language code (e.g., "hi" for Hindi)
            tgt: Target language code (e.g., "en" for English)
            max_pairs: Maximum number of sentence pairs to download

        Returns:
            Path to JSONL file with {"source_text": ..., "target_text": ..., "source_lang": ..., "target_lang": ...}
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("Install `datasets`: pip install datasets")
            raise

        output_dir = self.output_dir / "bpcc"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"bpcc_{src}_{tgt}.jsonl"

        logger.info(f"Fetching BPCC {src}-{tgt} | Max pairs: {max_pairs}")

        count = 0
        try:
            # BPCC is available as separate configs per language pair
            config_name = f"{src}-{tgt}"
            dataset = load_dataset(
                "ai4bharat/BPCC",
                config_name,
                split="train",
                streaming=True,
            )

            with open(output_path, "w", encoding="utf-8") as f:
                for sample in dataset:
                    src_text = str(sample.get("source", sample.get(src, ""))).strip()
                    tgt_text = str(sample.get("target", sample.get(tgt, ""))).strip()

                    if not src_text or not tgt_text:
                        continue

                    record = {
                        "source_text": src_text,
                        "target_text": tgt_text,
                        "source_lang": src,
                        "target_lang": tgt,
                        "source_dataset": "bpcc",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

                    if count >= max_pairs:
                        break
                    if count % 10000 == 0:
                        logger.info(f"  BPCC: {count} pairs fetched")

        except Exception as e:
            logger.error(f"Error fetching BPCC {src}-{tgt}: {e}")
            logger.info("Available configs may differ. Check HF dataset card for exact config names.")

        logger.success(f"BPCC fetch complete: {count} pairs | Output: {output_path}")
        return output_path

    # ─── Summary ──────────────────────────────────────────────────────────────

    def fetch_all(
        self,
        dialects: Optional[list[str]] = None,
        max_vaani: int = 5000,
        max_karya: int = 10000,
    ) -> dict[str, Path]:
        """Fetch all datasets. Returns dict of {name: output_path}."""
        results = {}
        results["vaani"] = self.fetch_vaani(dialects=dialects, max_samples_per_dialect=max_vaani)
        results["karya"] = self.fetch_karya(max_samples=max_karya)
        return results

    def report(self) -> dict:
        """Report on what's been downloaded."""
        stats = {}
        for dataset_dir in self.output_dir.iterdir():
            if dataset_dir.is_dir():
                jsonl_files = list(dataset_dir.glob("*.jsonl"))
                for jf in jsonl_files:
                    with open(jf) as f:
                        count = sum(1 for _ in f)
                    stats[jf.name] = count
        return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch datasets for Rajasthani Dialect AI")
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--dialects", nargs="+", default=ALL_DIALECTS)
    parser.add_argument("--max-vaani", type=int, default=5000)
    parser.add_argument("--max-karya", type=int, default=10000)
    parser.add_argument("--with-audio", action="store_true", default=True, help="Download audio files for Whisper fine-tuning (default: True)")
    parser.add_argument("--no-audio", action="store_false", dest="with_audio", help="Skip audio download (metadata only)")
    parser.add_argument("--bpcc", action="store_true", help="Also fetch BPCC parallel text for MT")
    parser.add_argument("--bpcc-pairs", type=int, default=100000, help="Max BPCC pairs to download")
    args = parser.parse_args()

    fetcher = DatasetFetcher(output_dir=args.output_dir)

    if args.with_audio:
        fetcher.fetch_vaani_with_audio(dialects=args.dialects, max_samples_per_dialect=args.max_vaani)
    else:
        fetcher.fetch_all(dialects=args.dialects, max_vaani=args.max_vaani, max_karya=args.max_karya)

    if args.bpcc:
        fetcher.fetch_bpcc_sample(src="hi", tgt="en", max_pairs=args.bpcc_pairs)

    print("\n📊 Download Report:")
    for name, count in fetcher.report().items():
        print(f"  {name}: {count} records")
