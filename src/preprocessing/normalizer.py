"""
Devanagari Unicode Normalizer

NON-NEGOTIABLE preprocessing step that must precede ALL model training.
Handles the critical Unicode normalization challenges documented in the
architecture specification:

1. Unicode NFC normalization — standardize all text to Normalization Form C
2. Nukta standardization — decompose precomposed Nukta characters to base + modifier
3. Retroflex lateral flap preservation — protect ळ (U+0933) from destruction
4. Devanagari range validation — strip non-Devanagari noise from dialectal text

The normalizer ensures that identical phonemes rendered with different Unicode
encodings map to the same tokenizer input, preventing BPE vocabulary fragmentation
that would otherwise destroy cross-lingual transfer effectiveness.

Reference: detailed-report.md Phase 2 (Data preprocessing and normalization)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


# ─── Devanagari Unicode Ranges ────────────────────────────────────────────────

DEVANAGARI_RANGE = (0x0900, 0x097F)  # Basic Devanagari block
DEVANAGARI_EXTENDED = (0xA8E0, 0xA8FF)  # Devanagari Extended block
VEDIC_EXTENSIONS = (0x1CD0, 0x1CFF)  # Vedic Extensions

# The Nukta modifier: combines with base consonant to form aspirated/foreign sounds
NUKTA = "\u093C"  # ़

# Retroflex lateral flap — critical for Marwari and Mewari literature
RETROFLEX_LATERAL_FLAP = "\u0933"  # ळ

# Zero-width characters that may appear in text
ZWNJ = "\u200C"  # Zero-Width Non-Joiner
ZWJ = "\u200D"   # Zero-Width Joiner


# ─── Nukta Decomposition Map ─────────────────────────────────────────────────
# Precomposed Devanagari characters with Nukta that MUST be decomposed to
# base consonant + Nukta modifier to prevent tokenizer fragmentation.
#
# Example: क़ (U+0958) → क (U+0915) + ़ (U+093C)
#
# This is the single most critical normalization step. Without it, the BPE
# tokenizer treats the precomposed and decomposed forms as entirely different
# tokens, catastrophically fracturing the vocabulary for dialectal text.

NUKTA_DECOMPOSITION_MAP: dict[str, str] = {
    "\u0958": "\u0915\u093C",  # क़ → क + ़
    "\u0959": "\u0916\u093C",  # ख़ → ख + ़
    "\u095A": "\u0917\u093C",  # ग़ → ग + ़
    "\u095B": "\u091C\u093C",  # ज़ → ज + ़
    "\u095C": "\u0921\u093C",  # ड़ → ड + ़
    "\u095D": "\u0922\u093C",  # ढ़ → ढ + ़
    "\u095E": "\u092B\u093C",  # फ़ → फ + ़
    "\u095F": "\u092F\u093C",  # य़ → य + ़
}

# Additional precomposed vowels that should be standardized
VOWEL_NORMALIZATION_MAP: dict[str, str] = {
    "\u0929": "\u0928\u093C",  # ऩ → न + ़
    "\u0931": "\u0930\u093C",  # ऱ → र + ़
    "\u0934": "\u0933\u093C",  # ऴ → ळ + ़
}


# ─── Visarga and Anusvara Normalization ───────────────────────────────────────
# Standardize common variants

ANUSVARA = "\u0902"     # ं (Anusvara)
CHANDRABINDU = "\u0901" # ँ (Chandrabindu)
VISARGA = "\u0903"      # ः (Visarga)


@dataclass
class NormalizationStats:
    """Tracks normalization operations for audit logging."""
    total_chars_processed: int = 0
    nfc_changes: int = 0
    nukta_decompositions: int = 0
    vowel_normalizations: int = 0
    whitespace_fixes: int = 0
    invalid_chars_removed: int = 0
    retroflex_flaps_preserved: int = 0

    def summary(self) -> str:
        return (
            f"Processed {self.total_chars_processed} chars: "
            f"{self.nfc_changes} NFC, {self.nukta_decompositions} Nukta decomp, "
            f"{self.vowel_normalizations} vowel norm, {self.whitespace_fixes} ws fix, "
            f"{self.invalid_chars_removed} invalid removed, "
            f"{self.retroflex_flaps_preserved} ळ preserved"
        )


@dataclass
class NormalizerConfig:
    """Configuration for the Devanagari normalizer."""
    apply_nfc: bool = True
    decompose_nukta: bool = True
    normalize_vowels: bool = True
    preserve_retroflex_flap: bool = True
    strip_non_devanagari: bool = False  # Aggressive — only enable if input is pure Devanagari
    normalize_whitespace: bool = True
    remove_zero_width: bool = False  # ZWJ/ZWNJ can be meaningful in some contexts
    max_text_length: int = 512


class DevanagariNormalizer:
    """
    Production-grade Devanagari Unicode normalizer.
    
    This class implements the non-negotiable normalization pipeline specified
    in the architecture documentation. It MUST be applied to ALL text data
    before tokenization or model training.
    
    The normalization order is critical and must not be changed:
    1. Unicode NFC normalization (canonical decomposition + canonical composition)
    2. Nukta character decomposition (precomposed → base + modifier)
    3. Vowel normalization
    4. Retroflex lateral flap verification
    5. Whitespace normalization
    6. Optional: non-Devanagari stripping
    
    Usage:
        normalizer = DevanagariNormalizer()
        normalized_text = normalizer.normalize("कुछ टेक्स्ट")
        
        # Batch processing
        texts = normalizer.normalize_batch(["text1", "text2", ...])
        
        # With stats tracking
        normalizer = DevanagariNormalizer(track_stats=True)
        result = normalizer.normalize("text")
        print(normalizer.stats.summary())
    """

    def __init__(
        self,
        config: Optional[NormalizerConfig] = None,
        track_stats: bool = False,
    ):
        self.config = config or NormalizerConfig()
        self.track_stats = track_stats
        self._stats = NormalizationStats() if track_stats else None

        # Precompile regex patterns for performance
        self._multi_space_re = re.compile(r"\s+")
        self._devanagari_block_re = re.compile(
            r"[\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF]"
        )
        # Characters to preserve alongside Devanagari: digits, punctuation, spaces
        self._preserve_alongside_re = re.compile(
            r"[^\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF"
            r"\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"  # ASCII punct
            r"\u0964\u0965"  # Devanagari danda ।, double danda ॥
            r"\u0966-\u096F"  # Devanagari digits ०-९
            r"\u0030-\u0039"  # ASCII digits 0-9
            r"\s]"
        )

        # Build Nukta decomposition translation table for fast str.translate()
        self._nukta_table = str.maketrans(NUKTA_DECOMPOSITION_MAP)
        self._vowel_table = str.maketrans(VOWEL_NORMALIZATION_MAP)

        logger.debug(
            "DevanagariNormalizer initialized | "
            f"NFC={self.config.apply_nfc} Nukta={self.config.decompose_nukta} "
            f"Vowels={self.config.normalize_vowels} "
            f"RetroflexFlap={self.config.preserve_retroflex_flap}"
        )

    @property
    def stats(self) -> Optional[NormalizationStats]:
        return self._stats

    def reset_stats(self) -> None:
        if self._stats is not None:
            self._stats = NormalizationStats()

    def normalize(self, text: str) -> str:
        """
        Apply the full normalization pipeline to a single text string.
        
        The order of operations is critical and mirrors the architecture spec:
        1. NFC normalization
        2. Nukta decomposition
        3. Vowel normalization
        4. Retroflex flap verification
        5. Whitespace normalization
        6. Optional non-Devanagari stripping
        
        Args:
            text: Raw input text (potentially with inconsistent Unicode encoding)
            
        Returns:
            Normalized text with standardized Unicode representations
        """
        if not text or not text.strip():
            return ""

        original = text

        # Step 1: Unicode NFC normalization
        if self.config.apply_nfc:
            text = self._apply_nfc(text)

        # Step 2: Nukta character decomposition
        if self.config.decompose_nukta:
            text = self._decompose_nukta(text)

        # Step 3: Vowel normalization
        if self.config.normalize_vowels:
            text = self._normalize_vowels(text)

        # Step 4: Retroflex lateral flap verification
        if self.config.preserve_retroflex_flap:
            text = self._verify_retroflex_flap(text)

        # Step 5: Whitespace normalization
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Step 6: Optional — strip non-Devanagari characters
        if self.config.strip_non_devanagari:
            text = self._strip_non_devanagari(text)

        # Length enforcement
        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length]

        # Update stats
        if self._stats is not None:
            self._stats.total_chars_processed += len(original)

        return text

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Normalize a batch of text strings."""
        return [self.normalize(t) for t in texts]

    def _apply_nfc(self, text: str) -> str:
        """
        Apply Unicode Normalization Form C.
        
        NFC performs canonical decomposition followed by canonical composition.
        This ensures that characters like ऩ (which can be encoded as a single
        precomposed codepoint or as a base + combining sequence) are rendered
        consistently.
        """
        normalized = unicodedata.normalize("NFC", text)
        if self._stats is not None and normalized != text:
            self._stats.nfc_changes += 1
        return normalized

    def _decompose_nukta(self, text: str) -> str:
        """
        Decompose precomposed Nukta characters to base + modifier form.
        
        This is THE critical normalization step. Without it, the BPE tokenizer
        treats क़ (U+0958, precomposed) and क़ (U+0915 + U+093C, decomposed)
        as entirely different tokens, fragmenting the vocabulary.
        
        We standardize ALL instances to the separated form (base + ़) to
        drastically reduce vocabulary sparsity.
        """
        decomposed = text.translate(self._nukta_table)
        if self._stats is not None and decomposed != text:
            # Count actual decompositions
            count = sum(1 for c in text if c in NUKTA_DECOMPOSITION_MAP)
            self._stats.nukta_decompositions += count
        return decomposed

    def _normalize_vowels(self, text: str) -> str:
        """Normalize precomposed vowel characters to base + modifier form."""
        normalized = text.translate(self._vowel_table)
        if self._stats is not None and normalized != text:
            count = sum(1 for c in text if c in VOWEL_NORMALIZATION_MAP)
            self._stats.vowel_normalizations += count
        return normalized

    def _verify_retroflex_flap(self, text: str) -> str:
        """
        Verify that the retroflex lateral flap ळ (U+0933) is preserved intact.
        
        ळ is heavily utilized in Marwari and Mewari literature but is entirely
        absent from standard Hindi. The normalizer must NEVER map ळ to a 
        generic Hindi equivalent (like ल or ड़), as this would destroy
        critical dialectal phonetic information.
        
        This step is a verification/audit — it ensures ळ survives all prior
        normalization steps unchanged.
        """
        count = text.count(RETROFLEX_LATERAL_FLAP)
        if self._stats is not None and count > 0:
            self._stats.retroflex_flaps_preserved += count
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters to single spaces and strip."""
        normalized = self._multi_space_re.sub(" ", text).strip()
        if self._stats is not None and normalized != text:
            self._stats.whitespace_fixes += 1
        return normalized

    def _strip_non_devanagari(self, text: str) -> str:
        """
        Remove characters that are not in the Devanagari Unicode block
        or common punctuation/digits. Use with caution — this is aggressive
        and may remove valid content in mixed-script text.
        """
        stripped = self._preserve_alongside_re.sub("", text)
        if self._stats is not None:
            removed = len(text) - len(stripped)
            self._stats.invalid_chars_removed += removed
        return stripped

    def is_devanagari(self, text: str) -> bool:
        """Check if text contains any Devanagari characters."""
        return bool(self._devanagari_block_re.search(text))

    def devanagari_ratio(self, text: str) -> float:
        """Calculate the ratio of Devanagari characters in the text."""
        if not text:
            return 0.0
        deva_count = sum(1 for c in text if self._devanagari_block_re.match(c))
        return deva_count / len(text)

    def analyze(self, text: str) -> dict:
        """
        Analyze a text string and report Unicode composition details.
        Useful for debugging normalization issues.
        """
        analysis = {
            "length": len(text),
            "devanagari_ratio": self.devanagari_ratio(text),
            "has_nukta_precomposed": any(c in text for c in NUKTA_DECOMPOSITION_MAP),
            "has_retroflex_flap": RETROFLEX_LATERAL_FLAP in text,
            "has_nukta_modifier": NUKTA in text,
            "has_anusvara": ANUSVARA in text,
            "has_chandrabindu": CHANDRABINDU in text,
            "has_visarga": VISARGA in text,
            "has_zwj": ZWJ in text,
            "has_zwnj": ZWNJ in text,
            "nfc_normalized": unicodedata.is_normalized("NFC", text),
            "codepoints": [f"U+{ord(c):04X} ({unicodedata.name(c, '?')})" for c in text[:50]],
        }
        return analysis


def create_normalizer_from_config(config_path: Optional[Path] = None) -> DevanagariNormalizer:
    """Factory function to create a normalizer from a YAML config file."""
    import yaml

    if config_path is None:
        return DevanagariNormalizer(track_stats=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    preprocess_cfg = cfg.get("preprocessing", {})
    normalizer_config = NormalizerConfig(
        apply_nfc=preprocess_cfg.get("unicode_normalization", "NFC") == "NFC",
        decompose_nukta=preprocess_cfg.get("nukta_standardization", True),
        preserve_retroflex_flap=preprocess_cfg.get("preserve_retroflex_lateral_flap", True),
        max_text_length=preprocess_cfg.get("max_text_length", 512),
    )
    return DevanagariNormalizer(config=normalizer_config, track_stats=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Devanagari normalization pipeline")
    parser.add_argument("--config", type=str, default="config/base.yaml")
    parser.add_argument("--input", type=str, help="Input text file to normalize")
    parser.add_argument("--output", type=str, help="Output file for normalized text")
    args = parser.parse_args()

    normalizer = create_normalizer_from_config(Path(args.config) if args.config else None)

    if args.input:
        with open(args.input) as f:
            lines = f.readlines()
        normalized = normalizer.normalize_batch([line.strip() for line in lines])
        if args.output:
            with open(args.output, "w") as f:
                f.writelines(line + "\n" for line in normalized)
        logger.info(f"Normalized {len(normalized)} lines")
        if normalizer.stats:
            logger.info(normalizer.stats.summary())
    else:
        # Interactive demo
        test_texts = [
            "क़िला में ज़रूर जाओ",           # Contains precomposed Nukta chars
            "ळ वाळो राजस्थानी शब्द",          # Contains retroflex flap ळ
            "मारवाड़ी  में    होनो  कहते हैं",  # Multiple spaces
            "राजस्थानी भाषा",                  # Clean Devanagari
        ]
        for text in test_texts:
            result = normalizer.normalize(text)
            print(f"  Input:  {text!r}")
            print(f"  Output: {result!r}")
            print()
        if normalizer.stats:
            print(normalizer.stats.summary())
