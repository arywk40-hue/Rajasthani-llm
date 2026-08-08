"""
Text Cleaner for Rajasthani Dialect AI

General-purpose text cleaning utilities that complement the core Devanagari
normalizer. Handles:
- Script validation and filtering
- Punctuation normalization (Devanagari danda, ASCII)
- Digit normalization (Devanagari ↔ ASCII)
- Sentence boundary detection
- Text quality scoring for data filtering
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger


class ScriptType(Enum):
    """Script classification for input text."""
    DEVANAGARI = "devanagari"
    LATIN = "latin"
    MIXED = "mixed"
    OTHER = "other"
    EMPTY = "empty"


# Devanagari digit mapping
DEVANAGARI_DIGITS = "०१२३४५६७८९"
ASCII_DIGITS = "0123456789"
_DEVA_TO_ASCII = str.maketrans(DEVANAGARI_DIGITS, ASCII_DIGITS)
_ASCII_TO_DEVA = str.maketrans(ASCII_DIGITS, DEVANAGARI_DIGITS)

# Devanagari punctuation
DANDA = "\u0964"         # ।
DOUBLE_DANDA = "\u0965"  # ॥
DEVANAGARI_ABBREVIATION = "\u0970"  # ॰

# Common noise patterns in crowdsourced transcriptions
NOISE_PATTERNS = [
    re.compile(r"\[.*?\]"),     # [noise], [laughter], etc.
    re.compile(r"\(.*?\)"),     # (inaudible), (unclear), etc.
    re.compile(r"<.*?>"),       # <unk>, <silence>, etc.
    re.compile(r"\{.*?\}"),     # {pause}, etc.
]

# URL and email patterns
URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)


@dataclass
class TextQualityScore:
    """Quality assessment of a text sample for training data filtering."""
    text: str
    length: int
    devanagari_ratio: float
    script_type: ScriptType
    has_noise_markers: bool
    has_urls: bool
    word_count: int
    avg_word_length: float
    is_viable: bool  # Whether this text should be included in training
    rejection_reasons: list[str]


class TextCleaner:
    """
    General text cleaning and validation for the preprocessing pipeline.
    
    Applied after DevanagariNormalizer to handle non-Unicode-specific
    cleaning tasks. Focuses on data quality and consistency.
    
    Usage:
        cleaner = TextCleaner()
        
        # Clean a single text
        cleaned = cleaner.clean("some text with    extra    spaces")
        
        # Assess quality for data filtering
        score = cleaner.assess_quality("text sample")
        if score.is_viable:
            training_data.append(score.text)
    """

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 512,
        min_devanagari_ratio: float = 0.3,
        normalize_digits: str = "keep",  # "ascii", "devanagari", or "keep"
        remove_noise_markers: bool = True,
        remove_urls: bool = True,
        min_words: int = 1,
        max_words: int = 200,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_devanagari_ratio = min_devanagari_ratio
        self.normalize_digits = normalize_digits
        self.remove_noise_markers = remove_noise_markers
        self.remove_urls = remove_urls
        self.min_words = min_words
        self.max_words = max_words

        # Precompile patterns
        self._multi_space_re = re.compile(r" {2,}")
        self._multi_newline_re = re.compile(r"\n{3,}")
        self._devanagari_re = re.compile(r"[\u0900-\u097F]")
        self._latin_re = re.compile(r"[a-zA-Z]")

        logger.debug(
            f"TextCleaner initialized | len=[{min_length},{max_length}] "
            f"deva_ratio>={min_devanagari_ratio} digits={normalize_digits}"
        )

    def clean(self, text: str) -> str:
        """
        Apply the full text cleaning pipeline.
        
        Steps:
        1. Remove noise markers ([noise], (inaudible), etc.)
        2. Remove URLs and emails
        3. Normalize punctuation
        4. Normalize digits (optional)
        5. Collapse whitespace
        6. Strip and enforce length limits
        """
        if not text or not text.strip():
            return ""

        # Step 1: Remove noise markers from crowdsourced data
        if self.remove_noise_markers:
            for pattern in NOISE_PATTERNS:
                text = pattern.sub("", text)

        # Step 2: Remove URLs
        if self.remove_urls:
            text = URL_PATTERN.sub("", text)

        # Step 3: Normalize punctuation
        text = self._normalize_punctuation(text)

        # Step 4: Normalize digits
        if self.normalize_digits == "ascii":
            text = text.translate(_DEVA_TO_ASCII)
        elif self.normalize_digits == "devanagari":
            text = text.translate(_ASCII_TO_DEVA)

        # Step 5: Collapse whitespace
        text = self._multi_space_re.sub(" ", text)
        text = self._multi_newline_re.sub("\n\n", text)

        # Step 6: Strip and enforce length
        text = text.strip()
        if len(text) > self.max_length:
            text = text[: self.max_length]

        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a batch of texts, filtering out empty results."""
        return [cleaned for t in texts if (cleaned := self.clean(t))]

    def detect_script(self, text: str) -> ScriptType:
        """Classify the primary script of the text."""
        if not text or not text.strip():
            return ScriptType.EMPTY

        deva_count = len(self._devanagari_re.findall(text))
        latin_count = len(self._latin_re.findall(text))
        total_alpha = deva_count + latin_count

        if total_alpha == 0:
            return ScriptType.OTHER

        deva_ratio = deva_count / total_alpha
        if deva_ratio > 0.8:
            return ScriptType.DEVANAGARI
        elif deva_ratio < 0.2:
            return ScriptType.LATIN
        else:
            return ScriptType.MIXED

    def assess_quality(self, text: str) -> TextQualityScore:
        """
        Assess text quality for training data inclusion/exclusion decisions.
        
        Returns a TextQualityScore with a boolean is_viable flag.
        Texts that fail quality checks should be excluded from training.
        """
        rejection_reasons = []
        cleaned = self.clean(text)

        length = len(cleaned)
        if length < self.min_length:
            rejection_reasons.append(f"Too short: {length} < {self.min_length}")
        if length > self.max_length:
            rejection_reasons.append(f"Too long: {length} > {self.max_length}")

        # Devanagari ratio check
        deva_chars = len(self._devanagari_re.findall(cleaned))
        deva_ratio = deva_chars / max(length, 1)
        if deva_ratio < self.min_devanagari_ratio:
            rejection_reasons.append(f"Low Devanagari ratio: {deva_ratio:.2f} < {self.min_devanagari_ratio}")

        script_type = self.detect_script(cleaned)

        # Noise check
        has_noise = any(p.search(text) for p in NOISE_PATTERNS)
        has_urls = bool(URL_PATTERN.search(text))

        # Word count
        words = cleaned.split()
        word_count = len(words)
        if word_count < self.min_words:
            rejection_reasons.append(f"Too few words: {word_count} < {self.min_words}")
        if word_count > self.max_words:
            rejection_reasons.append(f"Too many words: {word_count} > {self.max_words}")

        avg_word_len = sum(len(w) for w in words) / max(word_count, 1)

        return TextQualityScore(
            text=cleaned,
            length=length,
            devanagari_ratio=deva_ratio,
            script_type=script_type,
            has_noise_markers=has_noise,
            has_urls=has_urls,
            word_count=word_count,
            avg_word_length=avg_word_len,
            is_viable=len(rejection_reasons) == 0,
            rejection_reasons=rejection_reasons,
        )

    def filter_viable(self, texts: list[str]) -> tuple[list[str], list[TextQualityScore]]:
        """
        Filter a list of texts, returning only viable ones for training.
        Also returns the quality scores for all texts (for logging/auditing).
        """
        scores = [self.assess_quality(t) for t in texts]
        viable = [s.text for s in scores if s.is_viable]
        rejected = [s for s in scores if not s.is_viable]

        if rejected:
            logger.info(
                f"Filtered {len(rejected)}/{len(texts)} texts "
                f"({len(viable)} viable)"
            )

        return viable, scores

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks to consistent forms."""
        # Normalize various quote styles to standard quotes
        text = text.replace("\u201C", '"').replace("\u201D", '"')  # "Smart" quotes
        text = text.replace("\u2018", "'").replace("\u2019", "'")  # 'Smart' quotes
        text = text.replace("\u2013", "-").replace("\u2014", "-")  # En/em dash
        text = text.replace("\u2026", "...")  # Ellipsis

        # Ensure space after Devanagari dandas
        text = re.sub(rf"({DANDA}|{DOUBLE_DANDA})(\S)", r"\1 \2", text)

        return text

    def split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using Devanagari sentence boundaries.
        Uses danda (।) and double danda (॥) as primary delimiters,
        with fallback to period (.) and question/exclamation marks.
        """
        # Split on Devanagari dandas and standard punctuation
        pattern = rf"[{DANDA}{DOUBLE_DANDA}.!?]+"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def devanagari_word_count(self, text: str) -> int:
        """Count words that contain at least one Devanagari character."""
        words = text.split()
        return sum(1 for w in words if self._devanagari_re.search(w))
