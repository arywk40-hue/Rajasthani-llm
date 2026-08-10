"""
Data Schemas for Rajasthani Dialect AI

This module defines standard Pydantic schemas for structured inputs/outputs
across all subsystems (ASR, MT, TTS, and End-to-End Pipeline).
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class ASRDataRecord(BaseModel):
    """Schema for ASR speech-to-text data records."""
    id: str = Field(..., description="Unique identifier for the sample")
    audio_path: str = Field(..., description="Local absolute or portable relative path to the audio file")
    dialect: str = Field(..., description="Specific dialect name (e.g., marwari, mewari)")
    transcript: str = Field(..., description="Original raw transcript in Devanagari")
    normalized_text: Optional[str] = Field(None, description="Cleaned and Unicode normalized transcript")


class DialectTextRecord(BaseModel):
    """Schema for monolingual dialect text records."""
    id: str = Field(..., description="Unique identifier for the sample")
    text: str = Field(..., description="Raw text in Devanagari")
    dialect: str = Field(..., description="Specific dialect name")
    normalized_text: Optional[str] = Field(None, description="Normalized Devanagari text")


class MTParallelRecord(BaseModel):
    """Schema for Machine Translation parallel translation pairs."""
    id: str = Field(..., description="Unique identifier for the translation pair")
    source_text: str = Field(..., description="Input text in source language/dialect")
    target_text: str = Field(..., description="Reference translation in target language")
    source_lang: str = Field(..., description="Source language code (e.g. marwari, hin_Deva)")
    target_lang: str = Field(..., description="Target language code (e.g. eng_Latn, hin_Deva)")
    dialect: Optional[str] = Field(None, description="Rajasthani dialect identifier if applicable")


class TTSDataRecord(BaseModel):
    """Schema for Text-to-Speech synthesis inputs and outputs."""
    id: str = Field(..., description="Unique identifier for the request/sample")
    text: str = Field(..., description="Normalized text to synthesize")
    dialect: str = Field(..., description="Target dialect/pronunciation style")
    output_path: Optional[str] = Field(None, description="Output WAV file location")


class PipelineExecutionResult(BaseModel):
    """Schema for the full End-to-End Speech-to-Speech Translation pipeline result."""
    id: str = Field(..., description="Execution identifier")
    audio_path: str = Field(..., description="Input audio file path")
    dialect: str = Field(..., description="Detected or specified input dialect")
    transcript: str = Field(..., description="ASR transcribed raw text")
    normalized_text: str = Field(..., description="ASR transcribed normalized text")
    translation: str = Field(..., description="MT translated target text")
    output_wav_path: str = Field(..., description="TTS synthesized output audio file path")
    
    # Latency tracking (in milliseconds)
    asr_latency_ms: float = Field(..., description="ASR subsystem latency")
    mt_latency_ms: float = Field(..., description="MT subsystem latency")
    tts_latency_ms: float = Field(..., description="TTS subsystem latency")
    total_latency_ms: float = Field(..., description="Total pipeline latency")
    
    # Audio durations (in seconds)
    input_duration_sec: float = Field(..., description="Duration of input audio")
    output_duration_sec: float = Field(..., description="Duration of synthesized output audio")
