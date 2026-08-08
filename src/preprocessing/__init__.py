"""
Preprocessing module for Rajasthani Dialect AI.

Provides the non-negotiable Devanagari normalization pipeline that must be
executed before ANY model training. Handles:
- Unicode NFC normalization
- Nukta character standardization
- Retroflex lateral flap (ळ) preservation
- Dialect-specific phonological shift mapping
- General text cleaning and validation
"""

from src.preprocessing.normalizer import DevanagariNormalizer
from src.preprocessing.phonological_mapper import PhonologicalMapper
from src.preprocessing.text_cleaner import TextCleaner

__all__ = [
    "DevanagariNormalizer",
    "PhonologicalMapper",
    "TextCleaner",
]
