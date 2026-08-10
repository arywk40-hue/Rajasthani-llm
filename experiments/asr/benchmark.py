"""
ASR Benchmark — real inference over VAANI audio.

Reads a manifest of {audio_path, text, dialect} records, runs IndicWhisper over the
audio, and reports per-dialect CER/WER plus a measured real-time factor.

    python experiments/asr/benchmark.py \
        --manifest data/raw/vaani/vaani_audio_metadata.jsonl \
        --output_csv results/asr_results.csv

The manifest is what `DatasetFetcher.fetch_vaani_with_audio()` writes. If it is missing
or empty the script exits non-zero without writing a CSV — there is deliberately no
placeholder path, because a fabricated row is worse than no row.

An earlier version of this file hardcoded (hypothesis, reference) pairs in which the two
strings were identical, so WER and CER were 0.0 by construction, and reported RTF as the
literal 0.15. Nothing here is hardcoded now: every number comes from model output timed
against audio duration.
"""

import csv
import json
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

from loguru import logger

from src.evaluation.metrics import compute_cer, compute_wer

DIALECTS = ["marwari", "mewari", "dhundhari", "hadoti", "mewati", "bagri"]


def load_manifest(path: Path) -> list[dict]:
    """Read the fetcher's JSONL manifest, keeping only rows whose audio is on disk."""
    if not path.exists():
        logger.error(
            f"Manifest not found: {path}\n"
            "Fetch audio first:\n"
            "  python scripts/fetch_data.py --with-audio --max-vaani 200"
        )
        return []

    records, missing = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            audio_path = rec.get("audio_path", "")
            text = str(rec.get("text", "")).strip()
            if not audio_path or not text:
                continue
            if not Path(audio_path).exists():
                missing += 1
                continue
            records.append(rec)

    if missing:
        logger.warning(f"{missing} manifest rows reference audio files that are not on disk.")
    return records


def audio_duration_seconds(paths: list[str]) -> float:
    """Total duration of the given audio files, for the real-time factor."""
    try:
        import soundfile as sf
    except ImportError:
        logger.warning("soundfile not installed — RTF cannot be computed.")
        return 0.0

    total = 0.0
    for p in paths:
        try:
            info = sf.info(str(p))
            total += info.frames / info.samplerate
        except Exception as e:
            logger.warning(f"Could not read duration of {p}: {e}")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASR Benchmark across Rajasthani Dialects")
    parser.add_argument("--manifest", type=str, default="data/raw/vaani/vaani_audio_metadata.jsonl")
    parser.add_argument("--output_csv", type=str, default="results/asr_results.csv")
    parser.add_argument("--model", type=str, default=None, help="Key from WHISPER_MODELS or an HF id")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Cap samples per dialect (0 = no cap)")
    args = parser.parse_args()

    records = load_manifest(Path(args.manifest))
    if not records:
        logger.error("No usable audio. Nothing benchmarked, no CSV written.")
        return 1

    by_dialect: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_dialect[rec.get("dialect", "unknown")].append(rec)

    unrecognised = {d: len(v) for d, v in by_dialect.items() if d not in DIALECTS}
    if unrecognised:
        logger.warning(
            f"Manifest rows with a dialect outside the six targets are not benchmarked: "
            f"{unrecognised}"
        )

    if args.limit:
        by_dialect = {d: rs[: args.limit] for d, rs in by_dialect.items()}

    logger.info(
        f"Benchmarking {sum(len(v) for v in by_dialect.values())} samples across "
        f"{len(by_dialect)} dialects: { {d: len(v) for d, v in by_dialect.items()} }"
    )

    from src.asr.model import WhisperASR

    asr = WhisperASR(model_name=args.model) if args.model else WhisperASR()

    results = []
    for dialect in DIALECTS:
        rows = by_dialect.get(dialect, [])
        if not rows:
            logger.warning(f"No audio for {dialect} — omitted from the CSV rather than zero-filled.")
            continue

        audio_paths = [r["audio_path"] for r in rows]
        references = [r["text"] for r in rows]

        start = time.perf_counter()
        hypotheses = asr.transcribe(audio_paths, batch_size=args.batch_size)
        elapsed = time.perf_counter() - start

        cers = [compute_cer(h, r) for h, r in zip(hypotheses, references)]
        wers = [compute_wer(h, r) for h, r in zip(hypotheses, references)]
        n = len(cers)

        # An empty hypothesis scores CER 1.0, which is correct but indistinguishable from
        # a genuinely bad transcription. Surface the count so a load/decode failure is not
        # mistaken for model quality.
        empty_hyps = sum(1 for h in hypotheses if not h.strip())
        if empty_hyps:
            logger.warning(
                f"{dialect}: {empty_hyps}/{n} hypotheses came back empty. These score "
                "CER 1.0 and inflate the mean — check audio loading before reporting."
            )

        audio_secs = audio_duration_seconds(audio_paths)
        rtf = round(elapsed / audio_secs, 4) if audio_secs > 0 else ""

        results.append({
            "model": asr.model_id,
            "dialect": dialect,
            "samples": n,
            "empty_hypotheses": empty_hyps,
            "wer": round(sum(wers) / n, 4),
            "cer": round(sum(cers) / n, 4),
            "rtf": rtf,
            "audio_seconds": round(audio_secs, 1),
            "wall_seconds": round(elapsed, 1),
            "measured": "true",
        })
        logger.info(
            f"{dialect}: CER={results[-1]['cer']} WER={results[-1]['wer']} "
            f"RTF={rtf} over {n} samples"
        )

    if not results:
        logger.error("No dialect had usable audio. No CSV written.")
        return 1

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "dialect", "samples", "empty_hypotheses", "wer", "cer",
            "rtf", "audio_seconds", "wall_seconds", "measured",
        ])
        writer.writeheader()
        writer.writerows(results)

    covered = {r["dialect"] for r in results}
    if missing_dialects := [d for d in DIALECTS if d not in covered]:
        logger.warning(
            f"These dialects have no measurement: {missing_dialects}. "
            "Report coverage alongside any figure taken from this CSV."
        )

    logger.success(f"Wrote {len(results)} measured rows to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
