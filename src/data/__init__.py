"""
Data Ingestion and Corpus Building

This module handles the extraction, validation, and normalization of diverse
dataset formats (Parquet, TSV) used in the Rajasthani Dialect AI ecosystem.
It provides data loaders for VAANI, BPCC, LDC-IL, IndicTTS, and Karya datasets,
along with a unified CorpusBuilder that orchestrates the data fusion pipeline.
"""

from src.data.corpus_builder import CorpusBuilder
from src.data.loaders import BPCCLoader, IndicTTSLoader, KaryaLoader, LDCILoader, VaaniLoader

__all__ = [
    "CorpusBuilder",
    "VaaniLoader",
    "BPCCLoader",
    "IndicTTSLoader",
    "KaryaLoader",
    "LDCILoader",
]
