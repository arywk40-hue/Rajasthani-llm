"""
End-to-End Speech-to-Speech Translation (S2ST) Pipeline

Orchestrates the full cascaded pipeline:
Audio Input 
  → ASR (Whisper) 
  → Normalization & Cognate mapping (DialectNormalizer) 
  → Machine Translation (IndicTrans2) 
  → TTS (Meta VITS) 
  → Audio Output
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger

from src.asr.inference import ASRInference
from src.mt.pipeline import TranslationPipeline
from src.tts.synthesize import IndicTTSSynthesizer
from src.data.schemas import PipelineExecutionResult


class SpeechToSpeechPipeline:
    """Orchestrator for the cascaded ASR → MT → TTS pipeline."""

    def __init__(
        self,
        asr_model_name: str = "whisper-tiny",
        mt_model_name: str = "indic-indic-dist-320M",
        tts_model_name: str = "facebook/mms-tts-hin",
        device: str = "auto",
    ):
        logger.info("Initializing End-to-End Speech-to-Speech Translation Pipeline")
        self.asr = ASRInference(model_name=asr_model_name, device=device)
        self.mt = TranslationPipeline(model_name=mt_model_name)
        self.tts = IndicTTSSynthesizer(model_name=tts_model_name, device=device)
        logger.success("End-to-End S2ST Pipeline components initialized successfully")

    def process(
        self,
        audio_path: str | Path,
        dialect: str,
        target_lang: str = "hindi",
        output_dir: str | Path = "results/e2e/",
    ) -> PipelineExecutionResult:
        """
        Execute the cascaded pipeline on an input audio file.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not audio_path.exists():
            raise FileNotFoundError(f"Input audio not found: {audio_path}")

        # Measure input duration
        info = sf.info(str(audio_path))
        input_duration = info.duration

        start_total = time.perf_counter()

        # Step 1: Automatic Speech Recognition (ASR)
        logger.info(f"[E2E Pipeline] Step 1: Running ASR on {audio_path}")
        asr_start = time.perf_counter()
        asr_res = self.asr.transcribe(str(audio_path), language="hi")
        asr_latency = (time.perf_counter() - asr_start) * 1000
        raw_text = asr_res["transcript"]
        normalized_text = asr_res["normalized_text"]
        logger.info(f"  ASR Transcript: {normalized_text}")

        # Step 2: Machine Translation (MT)
        logger.info(f"[E2E Pipeline] Step 2: Translating text via Dialect-Aware MT ({dialect} → {target_lang})")
        mt_start = time.perf_counter()
        mt_res = self.mt.translate_dialect(normalized_text, dialect=dialect, target_lang=target_lang)
        mt_latency = (time.perf_counter() - mt_start) * 1000
        translation = mt_res["translated_text"]
        logger.info(f"  MT Translation: {translation}")

        # Step 3: Text-to-Speech (TTS)
        logger.info(f"[E2E Pipeline] Step 3: Synthesizing output audio via VITS TTS")
        tts_start = time.perf_counter()
        output_wav_path = output_dir / "output.wav"
        
        # Run local TTS synthesis
        self.tts.synthesize_to_file(translation, output_wav_path)
        tts_latency = (time.perf_counter() - tts_start) * 1000

        # Measure output duration
        out_info = sf.info(str(output_wav_path))
        output_duration = out_info.duration

        total_latency = (time.perf_counter() - start_total) * 1000

        # Construct and validate contract output using Pydantic schema
        result = PipelineExecutionResult(
            id=f"e2e_{dialect}_{int(time.time())}",
            audio_path=str(audio_path),
            dialect=dialect.lower().strip(),
            transcript=raw_text,
            normalized_text=normalized_text,
            translation=translation,
            output_wav_path=str(output_wav_path),
            asr_latency_ms=round(asr_latency, 2),
            mt_latency_ms=round(mt_latency, 2),
            tts_latency_ms=round(tts_latency, 2),
            total_latency_ms=round(total_latency, 2),
            input_duration_sec=round(input_duration, 2),
            output_duration_sec=round(output_duration, 2)
        )

        # Save intermediate results in output directory as requested
        import json
        with open(output_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump({"transcript": raw_text, "normalized_text": normalized_text}, f, ensure_ascii=False, indent=2)
            
        with open(output_dir / "translation.json", "w", encoding="utf-8") as f:
            json.dump({"original": normalized_text, "translation": translation, "dialect": dialect, "target_lang": target_lang}, f, ensure_ascii=False, indent=2)
            
        with open(output_dir / "pipeline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

        logger.success(f"[E2E Pipeline] Processing completed in {result.total_latency_ms:.1f}ms")
        return result
