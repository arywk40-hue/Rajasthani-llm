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

# ─── HuggingFace repo ids ─────────────────────────────────────────────────────
# VAANI_REPO is the full corpus (~31,000 hrs, audio + metadata). Only ~2,043 hrs carry
# transcripts; those live in VAANI_TRANSCRIPTION_REPO. ASR fine-tuning needs transcripts,
# so prefer the transcription part when a config exists there for the district.
VAANI_REPO = "ARTPARK-IISc/Vaani"
VAANI_TRANSCRIPTION_REPO = "ARTPARK-IISc/Vaani-transcription-part"
KARYA_REPO = "severo/speech-rj-hi"
BPCC_REPO = "ai4bharat/BPCC"


def get_dialect_for_district(district: str) -> str:
    """Map a district name to its primary dialect."""
    return DISTRICT_DIALECT_MAP.get(district.lower().strip(), "unknown")


def vaani_configs_for_dialects(
    dialects: list[str],
    repo_id: str = VAANI_REPO,
) -> list[str]:
    """
    Resolve VAANI config names for the requested dialects.

    Asks the Hub for the repo's real config list and keeps the ones whose district maps to
    a requested dialect. Falls back to constructing `Rajasthan_<District>` candidates if the
    listing is unavailable (offline, or `datasets` not installed).

    Earlier revisions hardcoded configs for marwari and dhundhari only, so a request for the
    other four dialects loaded nothing and still logged success.
    """
    wanted_districts = {
        district for district, dialect in DISTRICT_DIALECT_MAP.items()
        if dialect in dialects
    }

    def _district_of(config: str) -> str:
        # Configs are "<State>_<District>"; a district may itself contain underscores.
        _, _, tail = config.partition("_")
        return tail.replace("_", " ").lower().strip()

    try:
        from datasets import get_dataset_config_names

        available = get_dataset_config_names(repo_id)
        matched = sorted(c for c in available if _district_of(c) in wanted_districts)

        if matched:
            logger.info(
                f"Resolved {len(matched)} VAANI configs from {len(available)} published, "
                f"for dialects {dialects}"
            )
            missing = wanted_districts - {_district_of(c) for c in matched}
            if missing:
                logger.warning(
                    f"VAANI publishes no config for these districts: {sorted(missing)}. "
                    "Dialects resting solely on them cannot be fetched from VAANI."
                )
            return matched

        logger.warning(
            f"None of the {len(available)} published configs matched the target districts. "
            "Falling back to constructed names — verify these against the dataset card."
        )
    except Exception as e:
        logger.warning(f"Could not list configs for {repo_id} ({e}). Using constructed names.")

    return sorted(
        f"Rajasthan_{'_'.join(p.capitalize() for p in d.split())}" for d in wanted_districts
    )


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

        configs_to_load = vaani_configs_for_dialects(dialects)
        logger.info(f"Candidate VAANI configs ({len(configs_to_load)}): {configs_to_load}")
        loaded_configs, skipped_configs = [], []

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for config_name in configs_to_load:
                    logger.info(f"Streaming VAANI configuration: {config_name}")
                    try:
                        dataset = load_dataset(
                            VAANI_REPO,
                            config_name,
                            split=split,
                            streaming=True,
                        )
                        # Avoid audio decoding during metadata fetch
                        dataset = dataset.cast_column("audio", datasets.Audio(decode=False))
                        loaded_configs.append(config_name)
                    except Exception as e:
                        logger.warning(f"VAANI config {config_name} unavailable, skipping: {e}")
                        skipped_configs.append(config_name)
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

        if skipped_configs:
            logger.warning(
                f"{len(skipped_configs)} of {len(configs_to_load)} configs did not load: "
                f"{skipped_configs}"
            )

        if total == 0:
            logger.error(
                "VAANI fetch produced 0 samples. Nothing was written to "
                f"{output_path}. Do not treat this as a completed fetch — check the config "
                "names above against the dataset card and confirm HF access."
            )
        else:
            logger.success(
                f"VAANI fetch: {total} total samples across {len(loaded_configs)} configs | "
                f"Per-dialect: {dialect_counts} | Output: {output_path}"
            )
            empty = [d for d, c in dialect_counts.items() if c == 0]
            if empty:
                logger.warning(f"No samples collected for: {empty}")
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

        Two caveats that bound what this data can be used for:
          - This method writes text and metadata only. `audio` is cast with decode=False,
            so `audio_path` is a reference inside the HF cache, not a local WAV. Audio for
            fine-tuning still has to be materialised separately.
          - Karya does not partition by dialect; every row is labelled `rajasthani`. The
            text is standard Hindi register, so it is Rajasthani-accented Hindi *speech* —
            useful for ASR acoustics, not a source of dialect text for MT.
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
                KARYA_REPO,
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
        demo_fallback: bool = True,
    ) -> Path:
        """
        Fetch VAANI data WITH audio files for ASR.
        
        Implements:
        - Exponential backoff retries on network failure
        - Resumability by checking existing, uncorrupted audio on disk
        - Metadata validation and sample rate verification
        - Portable relative paths in the generated manifest
        - Duplicate transcript removal
        """
        try:
            from datasets import load_dataset
            import soundfile as sf
            import numpy as np
            import hashlib
            import time
        except ImportError:
            logger.error("Install required packages: pip install datasets soundfile numpy")
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
        seen_texts = set()

        # If file exists, pre-load existing texts to ensure resumability
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            seen_texts.add(rec["text"])
                            dialect_counts[rec["dialect"]] = dialect_counts.get(rec["dialect"], 0) + 1
                            total += 1
            except Exception as e:
                logger.warning(f"Error reading existing manifest: {e}. Starting fresh.")
                seen_texts.clear()
                dialect_counts = {d: 0 for d in dialects}
                total = 0

        configs_to_load = vaani_configs_for_dialects(dialects)
        logger.info(f"Candidate VAANI configs ({len(configs_to_load)}): {configs_to_load}")
        loaded_configs, skipped_configs = [], []

        staging_path = metadata_path.with_name(metadata_path.name + ".partial")

        try:
            with open(staging_path, "w", encoding="utf-8") as f:
                # Write back already cached records
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as mf:
                        for line in mf:
                            if line.strip():
                                f.write(line)

                for config_name in configs_to_load:
                    logger.info(f"Streaming VAANI with audio configuration: {config_name}")
                    dataset = None
                    
                    # Exponential backoff for loading the dataset config
                    retries = 0
                    max_retries = 5
                    backoff = 2.0
                    while retries <= max_retries:
                        try:
                            dataset = load_dataset(
                                VAANI_REPO,
                                config_name,
                                split="train",
                                streaming=True,
                            )
                            loaded_configs.append(config_name)
                            break
                        except Exception as e:
                            retries += 1
                            if retries > max_retries:
                                logger.error(f"VAANI config {config_name} failed to load after {max_retries} retries: {e}")
                                skipped_configs.append(config_name)
                                break
                            logger.warning(f"Failed loading {config_name}: {e}. Retrying in {backoff}s...")
                            time.sleep(backoff)
                            backoff *= 2

                    if dataset is None:
                        continue

                    # Stream dataset iteration with exponential backoff for network drops
                    dataset_iter = iter(dataset)
                    retries = 0
                    while True:
                        try:
                            sample = next(dataset_iter)
                        except StopIteration:
                            break
                        except Exception as e:
                            retries += 1
                            if retries > max_retries:
                                logger.error(f"Error reading stream from {config_name}: {e}. Stopping stream config.")
                                break
                            backoff_time = 2.0 * (2 ** (retries - 1))
                            logger.warning(f"Stream error: {e}. Retrying config stream in {backoff_time}s...")
                            time.sleep(backoff_time)
                            # Re-initialize the iterator and resume (skips already verified files)
                            try:
                                dataset = load_dataset(VAANI_REPO, config_name, split="train", streaming=True)
                                dataset_iter = iter(dataset)
                            except Exception as re_err:
                                logger.error(f"Failed to re-initialize dataset: {re_err}")
                            continue

                        retries = 0  # Reset retries on successful read

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

                        text = str(sample.get("transcript", sample.get("text", sample.get("sentence", "")))).strip()
                        audio = sample.get("audio", {})

                        if not text or text == "None" or not isinstance(audio, dict):
                            continue

                        if text in seen_texts:
                            continue

                        audio_array = audio.get("array")
                        sr = audio.get("sampling_rate", 16000)
                        if audio_array is None:
                            continue

                        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
                        audio_filename = f"{dialect}_{text_hash}.wav"
                        audio_path = audio_dir / audio_filename
                        portable_path = f"data/raw/vaani/audio/{audio_filename}"

                        # Validate audio data and skip if valid (resumable)
                        is_valid = False
                        if audio_path.exists():
                            try:
                                info = sf.info(str(audio_path))
                                if info.duration > 0 and info.samplerate == sr:
                                    is_valid = True
                            except Exception:
                                audio_path.unlink(missing_ok=True)

                        if not is_valid:
                            try:
                                sf.write(str(audio_path), np.array(audio_array), sr)
                            except Exception as e:
                                logger.warning(f"Could not save audio {audio_filename}: {e}")
                                continue

                        record = {
                            "id": f"vaani_{dialect}_{text_hash}",
                            "audio_path": portable_path,
                            "text": text,
                            "dialect": dialect,
                            "district": district,
                            "sample_rate": sr,
                            "source": "vaani",
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        seen_texts.add(text)
                        dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
                        total += 1

                        if total % 100 == 0:
                            logger.info(f"  Audio saved: {total} | {dialect_counts}")

        except Exception as e:
            logger.error(f"Error fetching VAANI audio: {e}")

        # Staging promotion
        if total > 0:
            staging_path.replace(metadata_path)
            logger.success(f"VAANI audio fetch complete: {total} total records | Output: {metadata_path}")
            return metadata_path
        else:
            staging_path.unlink(missing_ok=True)
            if demo_fallback:
                logger.warning("No real VAANI data could be fetched. Generating DEMO/SYNTHETIC dataset...")
                return self.generate_demo_data()
            else:
                logger.error("VAANI audio fetch failed and demo_fallback is disabled.")
                return metadata_path

    def generate_demo_data(self) -> Path:
        """
        Generate synthetic demo audio and manifest to enable zero-shot / smoke-test pipelines.
        Clearly tags source as 'demo_synthetic' and uses standard Devanagari sentences.
        """
        import soundfile as sf
        import numpy as np
        import hashlib

        demo_dir = self.output_dir / "vaani"
        demo_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = demo_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = demo_dir / "vaani_audio_metadata.jsonl"

        # 6 dialects, 2 sentences each
        demo_sentences = {
            "marwari": ["राम राम सा, अठै सब चोखो है", "थारो नाम कांई है सा"],
            "mewari": ["अणी तरफ आओ सा", "राम राम, कइ हाल चाल है"],
            "dhundhari": ["छोरा कठै जा रयो छै", "थाको नाम कांई छै"],
            "hadoti": ["काय हाल छै भाई", "अठै आओ सा"],
            "mewati": ["कहाँ जा रह्यो है रे", "राम राम, के हाल है"],
            "bagri": ["किन्नै जावैगा भाई", "थांरो के नाम है"],
        }

        # Generate a 1-second sine wave (16kHz) for each sentence as fallback
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone

        logger.info("Writing demo synthetic audio files and manifest...")
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            for dialect, texts in demo_sentences.items():
                for idx, text in enumerate(texts):
                    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
                    audio_filename = f"demo_{dialect}_{text_hash}.wav"
                    audio_path = audio_dir / audio_filename
                    
                    if not audio_path.exists():
                        sf.write(str(audio_path), dummy_audio, sr)
                    
                    record = {
                        "id": f"demo_{dialect}_{text_hash}",
                        "audio_path": f"data/raw/vaani/audio/{audio_filename}",
                        "text": text,
                        "dialect": dialect,
                        "district": "demo_district",
                        "sample_rate": sr,
                        "source": "demo_synthetic",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.success(f"Generated demo dataset with 12 records: {metadata_path}")
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
                BPCC_REPO,
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
