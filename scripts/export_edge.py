"""
Script to export and quantize models for Suno Sutra edge deployment.

Usage:
    python scripts/export_edge.py --component asr --quantization int8
    python scripts/export_edge.py --component mt --quantization fp16
    python scripts/export_edge.py --component all --quantization int8 --validate
"""

import argparse
from pathlib import Path

from loguru import logger

from src.asr.model import FastConformerASR
from src.mt.model import IndicTrans2MT
from src.tts.fastpitch import FastPitchAcoustic
from src.tts.hifigan import HiFiGANVocoder
from src.edge.quantizer import ModelQuantizer, QuantizationType
from src.edge.onnx_exporter import ONNXExporter


def export_asr(quantizer: ModelQuantizer, exporter: ONNXExporter, quant: str) -> list[Path]:
    """Export ASR model for edge."""
    logger.info("Exporting ASR (FastConformer)...")
    model = FastConformerASR()
    paths = []

    if quant == "int8":
        paths.append(quantizer.quantize_int8(model, "asr_fastconformer"))
    else:
        paths.append(quantizer.quantize_fp16(model, "asr_fastconformer"))

    return paths


def export_mt(quantizer: ModelQuantizer, exporter: ONNXExporter, quant: str) -> list[Path]:
    """Export MT model for edge."""
    logger.info("Exporting MT (IndicTrans2)...")
    model = IndicTrans2MT()
    paths = []

    if quant == "int8":
        paths.append(quantizer.quantize_int8(model, "mt_indictrans2"))
    else:
        paths.append(quantizer.quantize_fp16(model, "mt_indictrans2"))

    # Also export to CTranslate2 for optimized NMT inference
    ct2_path = exporter.export_mt_to_ctranslate2(
        "mt_indictrans2", quantization=quant
    )
    paths.append(ct2_path)

    return paths


def export_tts(quantizer: ModelQuantizer, exporter: ONNXExporter, quant: str) -> list[Path]:
    """Export TTS models for edge."""
    logger.info("Exporting TTS (FastPitch + HiFi-GAN)...")
    acoustic = FastPitchAcoustic()
    vocoder = HiFiGANVocoder()
    paths = []

    if quant == "int8":
        paths.append(quantizer.quantize_int8(acoustic, "tts_fastpitch"))
        paths.append(quantizer.quantize_int8(vocoder, "tts_hifigan"))
    else:
        paths.append(quantizer.quantize_fp16(acoustic, "tts_fastpitch"))
        paths.append(quantizer.quantize_fp16(vocoder, "tts_hifigan"))

    return paths


def main():
    parser = argparse.ArgumentParser(description="Export models for Suno Sutra edge deployment")
    parser.add_argument("--component", choices=["asr", "mt", "tts", "all"], default="all")
    parser.add_argument("--quantization", choices=["int8", "fp16"], default="int8")
    parser.add_argument("--output", type=str, default="models/edge")
    parser.add_argument("--validate", action="store_true", help="Validate edge memory budget")
    args = parser.parse_args()

    quantizer = ModelQuantizer(output_dir=Path(args.output) / "quantized")
    exporter = ONNXExporter(output_dir=Path(args.output) / "onnx")

    all_paths: list[Path] = []

    if args.component in ("asr", "all"):
        all_paths.extend(export_asr(quantizer, exporter, args.quantization))
    if args.component in ("mt", "all"):
        all_paths.extend(export_mt(quantizer, exporter, args.quantization))
    if args.component in ("tts", "all"):
        all_paths.extend(export_tts(quantizer, exporter, args.quantization))

    if args.validate:
        # Only validate actual files (not directories like CT2 output)
        file_paths = [p for p in all_paths if p.is_file()]
        quantizer.validate_edge_budget(*file_paths)

    logger.info("Edge export pipeline complete.")


if __name__ == "__main__":
    main()
