"""
Tests for the Devanagari normalization pipeline.

These tests validate the non-negotiable preprocessing steps documented
in the architecture specification, with focus on:
- Unicode NFC normalization
- Nukta character decomposition
- Retroflex lateral flap (ळ) preservation
- Whitespace normalization
- Phonological mapper analysis
- Text cleaner quality scoring
"""

import pytest

from src.preprocessing.normalizer import (
    NUKTA,
    NUKTA_DECOMPOSITION_MAP,
    RETROFLEX_LATERAL_FLAP,
    DevanagariNormalizer,
    NormalizerConfig,
)
from src.preprocessing.phonological_mapper import (
    Dialect,
    PhonologicalMapper,
)
from src.preprocessing.text_cleaner import (
    ScriptType,
    TextCleaner,
)


# ─── DevanagariNormalizer Tests ──────────────────────────────────────────────


class TestDevanagariNormalizer:
    """Test suite for the core Unicode normalizer."""

    def setup_method(self) -> None:
        self.normalizer = DevanagariNormalizer(track_stats=True)

    def test_empty_input(self) -> None:
        assert self.normalizer.normalize("") == ""
        assert self.normalizer.normalize("   ") == ""
        assert self.normalizer.normalize("\n\t") == ""

    def test_clean_devanagari_passthrough(self) -> None:
        """Clean Devanagari text should pass through unchanged."""
        text = "राजस्थानी भाषा"
        assert self.normalizer.normalize(text) == text

    def test_nukta_decomposition_single(self) -> None:
        """Precomposed Nukta characters must decompose to base + modifier."""
        # Disable NFC to test decompose_nukta in isolation
        config = NormalizerConfig(apply_nfc=False)
        normalizer = DevanagariNormalizer(config=config, track_stats=True)
        # क़ (U+0958) → क (U+0915) + ़ (U+093C)
        precomposed = "\u0958"
        expected = "\u0915\u093C"
        result = normalizer.normalize(precomposed)
        assert result == expected
        assert normalizer.stats.nukta_decompositions == 1

    def test_nukta_decomposition_all_characters(self) -> None:
        """All precomposed Nukta characters must decompose correctly."""
        config = NormalizerConfig(apply_nfc=False)
        normalizer = DevanagariNormalizer(config=config)
        for precomposed, decomposed in NUKTA_DECOMPOSITION_MAP.items():
            result = normalizer.normalize(precomposed)
            assert result == decomposed, (
                f"Failed for U+{ord(precomposed):04X}: "
                f"expected {decomposed!r}, got {result!r}"
            )

    def test_nukta_decomposition_in_context(self) -> None:
        """Nukta decomposition must work within full words/sentences."""
        # "क़िला" with precomposed क़
        text = "\u0958\u093F\u0932\u093E"
        result = self.normalizer.normalize(text)
        # Should decompose to क + ़ + ि + ला
        assert "\u0958" not in result
        assert "\u0915" in result  # Base क present
        assert NUKTA in result     # Nukta modifier present

    def test_nukta_multiple_in_sentence(self) -> None:
        """Multiple Nukta chars in one sentence must all decompose."""
        config = NormalizerConfig(apply_nfc=False)
        normalizer = DevanagariNormalizer(config=config, track_stats=True)
        # "क़िला में ज़रूर" — two precomposed Nukta chars
        text = "\u0958\u093F\u0932\u093E \u092E\u0947\u0902 \u095B\u0930\u0942\u0930"
        result = normalizer.normalize(text)
        assert "\u0958" not in result  # No precomposed क़
        assert "\u095B" not in result  # No precomposed ज़
        assert normalizer.stats.nukta_decompositions == 2

    def test_retroflex_lateral_flap_preserved(self) -> None:
        """ळ (U+0933) must survive normalization intact."""
        text = f"वा{RETROFLEX_LATERAL_FLAP}ो राजस्थानी"
        result = self.normalizer.normalize(text)
        assert RETROFLEX_LATERAL_FLAP in result
        assert result.count(RETROFLEX_LATERAL_FLAP) == text.count(RETROFLEX_LATERAL_FLAP)
        assert self.normalizer.stats.retroflex_flaps_preserved == 1

    def test_retroflex_flap_with_nukta(self) -> None:
        """ळ must be preserved even when Nukta decomposition is active."""
        text = f"\u0958 {RETROFLEX_LATERAL_FLAP} \u095B"
        result = self.normalizer.normalize(text)
        assert RETROFLEX_LATERAL_FLAP in result
        assert "\u0958" not in result  # Nukta still decomposed

    def test_whitespace_normalization(self) -> None:
        """Multiple spaces must collapse to single space."""
        text = "मारवाड़ी  में    होनो  कहते हैं"
        result = self.normalizer.normalize(text)
        assert "  " not in result
        assert result == "मारवाड़ी में होनो कहते हैं"

    def test_leading_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace must be stripped."""
        text = "   राजस्थानी   "
        assert self.normalizer.normalize(text) == "राजस्थानी"

    def test_nfc_normalization(self) -> None:
        """NFC normalization must compose compatible characters."""
        import unicodedata
        text = "राजस्थानी"
        result = self.normalizer.normalize(text)
        assert unicodedata.is_normalized("NFC", result)

    def test_max_length_enforcement(self) -> None:
        """Text exceeding max_length must be truncated."""
        config = NormalizerConfig(max_text_length=10)
        normalizer = DevanagariNormalizer(config=config)
        text = "क" * 20
        result = normalizer.normalize(text)
        assert len(result) <= 10

    def test_strip_non_devanagari(self) -> None:
        """When enabled, non-Devanagari chars should be stripped."""
        config = NormalizerConfig(strip_non_devanagari=True)
        normalizer = DevanagariNormalizer(config=config, track_stats=True)
        text = "राजस्थानी text with English"
        result = normalizer.normalize(text)
        assert "text" not in result
        assert "राजस्थानी" in result

    def test_is_devanagari(self) -> None:
        assert self.normalizer.is_devanagari("राजस्थानी") is True
        assert self.normalizer.is_devanagari("English only") is False
        assert self.normalizer.is_devanagari("mixed मिश्रित") is True
        assert self.normalizer.is_devanagari("") is False

    def test_devanagari_ratio(self) -> None:
        assert self.normalizer.devanagari_ratio("राजस्थानी") > 0.9
        assert self.normalizer.devanagari_ratio("English") == 0.0
        assert 0.0 < self.normalizer.devanagari_ratio("mix मिश्रित") < 1.0

    def test_analyze(self) -> None:
        """Analysis should correctly identify Unicode composition."""
        analysis = self.normalizer.analyze("\u0958 ळ")
        assert analysis["has_nukta_precomposed"] is True
        assert analysis["has_retroflex_flap"] is True

    def test_batch_normalization(self) -> None:
        texts = ["राजस्थानी  भाषा", "\u0958\u093F\u0932\u093E", "  "]
        results = self.normalizer.normalize_batch(texts)
        assert len(results) == 3
        assert results[0] == "राजस्थानी भाषा"
        assert "\u0958" not in results[1]
        assert results[2] == ""

    def test_stats_tracking(self) -> None:
        """Stats must accurately track normalization operations."""
        config = NormalizerConfig(apply_nfc=False)
        normalizer = DevanagariNormalizer(config=config, track_stats=True)
        normalizer.normalize("\u0958  ळ")
        stats = normalizer.stats
        assert stats is not None
        assert stats.total_chars_processed > 0
        assert stats.nukta_decompositions == 1
        assert stats.retroflex_flaps_preserved == 1
        assert stats.whitespace_fixes == 1

    def test_stats_reset(self) -> None:
        normalizer = DevanagariNormalizer(track_stats=True)
        normalizer.normalize("test")
        normalizer.reset_stats()
        assert normalizer.stats.total_chars_processed == 0

    def test_idempotent(self) -> None:
        """Normalizing already-normalized text should be a no-op."""
        text = "राजस्थानी भाषा में ळ"
        first = self.normalizer.normalize(text)
        second = self.normalizer.normalize(first)
        assert first == second


# ─── PhonologicalMapper Tests ────────────────────────────────────────────────


class TestPhonologicalMapper:
    """Test suite for the phonological mapper."""

    def setup_method(self) -> None:
        self.mapper = PhonologicalMapper()

    def test_get_marwari_rules(self) -> None:
        rules = self.mapper.get_rules(Dialect.MARWARI)
        assert len(rules) > 0
        rule_names = [r.name for r in rules]
        assert "s_to_h_shift" in rule_names
        assert "retroflex_lateral_flap" in rule_names

    def test_get_mewari_rules(self) -> None:
        rules = self.mapper.get_rules(Dialect.MEWARI)
        assert len(rules) > 0

    def test_empty_rules_for_unvalidated_dialects(self) -> None:
        """Hadoti and Mewati have empty rules pending linguist validation."""
        assert len(self.mapper.get_rules(Dialect.HADOTI)) == 0
        assert len(self.mapper.get_rules(Dialect.MEWATI)) == 0

    def test_analyze_marwari_text(self) -> None:
        # Text with ळ (retroflex flap) — a Marwari marker
        text = "ळ वाळो शब्द"
        analysis = self.mapper.analyze(text, Dialect.MARWARI)
        assert analysis["dialect"] == "marwari"
        assert analysis["has_dialectal_features"] is True
        assert len(analysis["active_rules"]) > 0

    def test_analyze_plain_hindi(self) -> None:
        """Plain Hindi text should show fewer dialectal features."""
        text = "यह एक सामान्य हिंदी वाक्य है"
        analysis = self.mapper.analyze(text, Dialect.MARWARI)
        # Hindi /s/ (स) triggers the s_to_h bidirectional rule
        assert analysis["dialect"] == "marwari"

    def test_annotate(self) -> None:
        text = "ळ"
        annotation = self.mapper.annotate(text, Dialect.MARWARI)
        assert annotation["text"] == text
        assert annotation["dialect"] == "marwari"
        assert "analysis" in annotation
        assert "char_annotations" in annotation

    def test_detect_dialect(self) -> None:
        # Text with strong Marwari marker (ळ)
        text = "ळ वाळो मारवाड़ी"
        results = self.mapper.detect_dialect(text)
        assert len(results) > 0
        # Marwari and Mewari should rank high (both use ळ)
        top_dialects = [d.value for d, _ in results[:2]]
        assert "marwari" in top_dialects or "mewari" in top_dialects

    def test_hindi_cognate_hints(self) -> None:
        text = "ळ"
        hints = self.mapper.get_hindi_cognate_hints(text, Dialect.MARWARI)
        assert len(hints) > 0
        assert any(h["rule"] == "retroflex_lateral_flap" for h in hints)

    def test_rules_sorted_by_priority(self) -> None:
        """Rules should be returned sorted by priority (highest first)."""
        rules = self.mapper.get_rules(Dialect.MARWARI)
        priorities = [r.priority for r in rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_list_all_rules(self) -> None:
        all_rules = PhonologicalMapper.list_all_rules()
        assert "marwari" in all_rules
        assert "mewari" in all_rules
        assert len(all_rules["marwari"]) > 0


# ─── TextCleaner Tests ───────────────────────────────────────────────────────


class TestTextCleaner:
    """Test suite for the text cleaner."""

    def setup_method(self) -> None:
        self.cleaner = TextCleaner()

    def test_empty_input(self) -> None:
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean("   ") == ""

    def test_clean_passthrough(self) -> None:
        text = "राजस्थानी भाषा"
        assert self.cleaner.clean(text) == text

    def test_remove_noise_markers(self) -> None:
        text = "राजस्थानी [noise] भाषा (inaudible) है"
        result = self.cleaner.clean(text)
        assert "[noise]" not in result
        assert "(inaudible)" not in result
        assert "राजस्थानी" in result

    def test_remove_urls(self) -> None:
        text = "visit https://example.com for more"
        result = self.cleaner.clean(text)
        assert "https://example.com" not in result

    def test_digit_normalization_ascii(self) -> None:
        cleaner = TextCleaner(normalize_digits="ascii")
        text = "१२३ test"
        result = cleaner.clean(text)
        assert "123" in result

    def test_digit_normalization_devanagari(self) -> None:
        cleaner = TextCleaner(normalize_digits="devanagari")
        text = "123 test"
        result = cleaner.clean(text)
        assert "१२३" in result

    def test_script_detection_devanagari(self) -> None:
        assert self.cleaner.detect_script("राजस्थानी") == ScriptType.DEVANAGARI

    def test_script_detection_latin(self) -> None:
        assert self.cleaner.detect_script("English only") == ScriptType.LATIN

    def test_script_detection_mixed(self) -> None:
        assert self.cleaner.detect_script("mixed मिश्रित text") == ScriptType.MIXED

    def test_script_detection_empty(self) -> None:
        assert self.cleaner.detect_script("") == ScriptType.EMPTY

    def test_quality_assessment_viable(self) -> None:
        score = self.cleaner.assess_quality("राजस्थानी भाषा बहुत सुंदर है")
        assert score.is_viable is True
        assert score.devanagari_ratio > 0.5
        assert len(score.rejection_reasons) == 0

    def test_quality_assessment_too_short(self) -> None:
        cleaner = TextCleaner(min_length=10)
        score = cleaner.assess_quality("ही")
        assert score.is_viable is False
        assert any("short" in r.lower() for r in score.rejection_reasons)

    def test_quality_assessment_low_devanagari(self) -> None:
        score = self.cleaner.assess_quality("This is all English text here")
        assert score.is_viable is False

    def test_filter_viable(self) -> None:
        texts = [
            "राजस्थानी भाषा बहुत सुंदर है",
            "",
            "This is English",
            "मारवाड़ी में बोला जाता है",
        ]
        viable, scores = self.cleaner.filter_viable(texts)
        assert len(viable) >= 2  # At least the two Devanagari texts
        assert len(scores) == len(texts)

    def test_sentence_splitting(self) -> None:
        text = "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।"
        sentences = self.cleaner.split_sentences(text)
        assert len(sentences) == 3

    def test_smart_quotes_normalization(self) -> None:
        text = "\u201Cस्मार्ट\u201D quotes"
        result = self.cleaner.clean(text)
        assert "\u201C" not in result
        assert "\u201D" not in result

    def test_batch_cleaning(self) -> None:
        texts = ["राजस्थानी भाषा", "", "   ", "मारवाड़ी"]
        results = self.cleaner.clean_batch(texts)
        # Empty strings should be filtered out
        assert "" not in results
        assert len(results) == 2
