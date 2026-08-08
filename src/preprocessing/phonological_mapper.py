"""
Phonological Mapper for Rajasthani Dialects

Implements dialect-specific phonological shift rules that encode the systematic
sound changes between standard Hindi and Rajasthani dialects. These mappings
are applied as structured rules (not free-text edits) per the architecture spec.

Key phonological phenomena handled:
- Hindi /s/ → Marwari /h/ (e.g., Hindi "sona" → Marwari "hono")
- Okarant → Akarant pluralization shifts
- Dialect-specific vowel mutations in verb conjugations
- Retroflex consonant patterns unique to specific dialects

Reference: detailed-report.md Phase 2 (Phonological preservation algorithms)
Reference: architecture-documenattion.md (Linguistic Rules Integration)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


class Dialect(Enum):
    """Supported Rajasthani dialects."""
    MARWARI = "marwari"
    MEWARI = "mewari"
    DHUNDHARI = "dhundhari"
    HADOTI = "hadoti"
    MEWATI = "mewati"
    BAGRI = "bagri"
    HINDI = "hindi"  # Anchor language for reference


@dataclass(frozen=True)
class PhonologicalRule:
    """
    A single phonological mapping rule.
    
    Rules encode systematic sound shifts between Hindi and a target dialect.
    They are applied as structured transformations, never as free-text edits.
    
    Attributes:
        name: Human-readable name for the rule
        source_dialect: The dialect this sound appears in
        source_pattern: The phonological pattern in the source
        target_pattern: What it maps to in Hindi (or vice versa)
        context: Linguistic context where this rule applies
        bidirectional: Whether the rule applies in both directions
        priority: Higher priority rules are applied first (for conflict resolution)
    """
    name: str
    source_dialect: Dialect
    source_pattern: str
    target_pattern: str
    context: str = ""
    bidirectional: bool = False
    priority: int = 0
    regex_pattern: Optional[str] = None


# ─── Phonological Rule Database ──────────────────────────────────────────────
# These rules encode the documented phonological shifts between Hindi and
# Rajasthani dialects. Each rule is traceable to the linguistic documentation.

MARWARI_RULES: list[PhonologicalRule] = [
    PhonologicalRule(
        name="s_to_h_shift",
        source_dialect=Dialect.MARWARI,
        source_pattern="ह",  # Marwari /h/
        target_pattern="स",  # Hindi /s/
        context="Word-initial and medial positions. Hindi /s/ systematically becomes /h/ in Marwari.",
        bidirectional=True,
        priority=10,
    ),
    PhonologicalRule(
        name="okarant_to_akarant_plural",
        source_dialect=Dialect.MARWARI,
        source_pattern="ा",  # Akarant (plural ending)
        target_pattern="ो",  # Okarant (singular ending)
        context="Singular nouns ending in okarant (e.g., घरो) shift to akarant (घरा) in plural. Diverges from standard Hindi conventions.",
        bidirectional=False,
        priority=5,
    ),
    PhonologicalRule(
        name="retroflex_lateral_flap",
        source_dialect=Dialect.MARWARI,
        source_pattern="ळ",  # ळ (U+0933) — retroflex lateral flap
        target_pattern="ल",  # ल (U+0932) — dental lateral in Hindi
        context="Retroflex ळ is heavily used in Marwari/Mewari literature. Absent in standard Hindi which uses ल instead.",
        bidirectional=True,
        priority=15,
    ),
    PhonologicalRule(
        name="nasalization_pattern",
        source_dialect=Dialect.MARWARI,
        source_pattern="ँ",  # Chandrabindu (nasalized)
        target_pattern="ं",  # Anusvara
        context="Marwari uses chandrabindu more extensively for nasalization where Hindi would use anusvara.",
        bidirectional=True,
        priority=3,
    ),
]

MEWARI_RULES: list[PhonologicalRule] = [
    PhonologicalRule(
        name="mewari_a_vowel_shift",
        source_dialect=Dialect.MEWARI,
        source_pattern="ा",  # Specific A vowel sounds
        target_pattern="ा",
        context="Mewari relies heavily on specific A and O vowel sounds during verb conjugations, creating distinct prosodic patterns.",
        bidirectional=False,
        priority=5,
    ),
    PhonologicalRule(
        name="mewari_retroflex_lateral",
        source_dialect=Dialect.MEWARI,
        source_pattern="ळ",
        target_pattern="ल",
        context="Like Marwari, Mewari literature heavily utilizes the retroflex lateral flap ळ.",
        bidirectional=True,
        priority=15,
    ),
]

DHUNDHARI_RULES: list[PhonologicalRule] = [
    PhonologicalRule(
        name="dhundhari_aspirate_shift",
        source_dialect=Dialect.DHUNDHARI,
        source_pattern="ह",
        target_pattern="स",
        context="Dhundhari (Eastern Rajasthan primary dialect) shares the /s/→/h/ shift with Marwari but with different distribution.",
        bidirectional=True,
        priority=8,
    ),
]

BAGRI_RULES: list[PhonologicalRule] = [
    PhonologicalRule(
        name="bagri_gujarati_influence",
        source_dialect=Dialect.BAGRI,
        source_pattern="",
        target_pattern="",
        context="Bagri exhibits strong phonetic influences from neighboring Gujarati. Specific mappings require linguist validation on real data.",
        bidirectional=False,
        priority=1,
    ),
]

# Consolidated rule registry
DIALECT_RULES: dict[Dialect, list[PhonologicalRule]] = {
    Dialect.MARWARI: MARWARI_RULES,
    Dialect.MEWARI: MEWARI_RULES,
    Dialect.DHUNDHARI: DHUNDHARI_RULES,
    Dialect.HADOTI: [],  # Rules to be added after linguist validation
    Dialect.MEWATI: [],  # Rules to be added after linguist validation
    Dialect.BAGRI: BAGRI_RULES,
}


@dataclass
class MappingResult:
    """Result of a phonological mapping operation."""
    original: str
    mapped: str
    rules_applied: list[str] = field(default_factory=list)
    dialect: Optional[Dialect] = None


class PhonologicalMapper:
    """
    Maps dialect-specific phonological features to/from standard Hindi.
    
    This mapper does NOT perform automatic text transformation by default.
    Instead, it provides:
    
    1. Annotation: Tags text with phonological features for downstream models
    2. Analysis: Reports which phonological rules are active in a text
    3. Controlled mapping: Applies specific rules when explicitly requested
    
    The reason for this conservative approach: automatic phonological mapping
    during preprocessing could destroy dialectal information that the MT model
    needs to learn. The mapper is primarily used for:
    - Feature extraction (feeding phonological annotations to the model)
    - Evaluation (comparing model outputs against expected phonological patterns)
    - Data validation (ensuring training data preserves dialectal features)
    
    Usage:
        mapper = PhonologicalMapper()
        
        # Analyze text for phonological features
        features = mapper.analyze("होनो कहते हैं", Dialect.MARWARI)
        
        # Get applicable rules for a dialect
        rules = mapper.get_rules(Dialect.MARWARI)
        
        # Annotate text with phonological metadata
        annotations = mapper.annotate("text", Dialect.MARWARI)
    """

    def __init__(self, custom_rules: Optional[dict[Dialect, list[PhonologicalRule]]] = None):
        self._rules = {**DIALECT_RULES}
        if custom_rules:
            for dialect, rules in custom_rules.items():
                self._rules.setdefault(dialect, []).extend(rules)

        # Sort rules by priority (highest first) for each dialect
        for dialect in self._rules:
            self._rules[dialect] = sorted(
                self._rules[dialect], key=lambda r: r.priority, reverse=True
            )

        total_rules = sum(len(r) for r in self._rules.values())
        logger.debug(f"PhonologicalMapper initialized with {total_rules} rules across {len(self._rules)} dialects")

    def get_rules(self, dialect: Dialect) -> list[PhonologicalRule]:
        """Get all phonological rules for a specific dialect."""
        return self._rules.get(dialect, [])

    def analyze(self, text: str, dialect: Dialect) -> dict:
        """
        Analyze text for active phonological features of a specific dialect.
        
        Returns a dict with:
        - active_rules: Rules whose patterns are found in the text
        - feature_counts: Count of each phonological feature found
        - dialectal_markers: Characters/patterns that mark this as dialectal text
        """
        rules = self.get_rules(dialect)
        active_rules = []
        feature_counts = {}
        dialectal_markers = []

        for rule in rules:
            if not rule.source_pattern:
                continue

            count = text.count(rule.source_pattern)
            if count > 0:
                active_rules.append(rule.name)
                feature_counts[rule.name] = count

                if rule.source_pattern not in [r.target_pattern for r in rules]:
                    dialectal_markers.append({
                        "pattern": rule.source_pattern,
                        "rule": rule.name,
                        "context": rule.context,
                    })

        # Check for retroflex lateral flap specifically (critical marker)
        retroflex_count = text.count("ळ")
        if retroflex_count > 0 and dialect in (Dialect.MARWARI, Dialect.MEWARI):
            if "retroflex_lateral_flap" not in active_rules:
                active_rules.append("retroflex_lateral_flap_detected")
                feature_counts["retroflex_lateral_flap_detected"] = retroflex_count

        return {
            "dialect": dialect.value,
            "text_length": len(text),
            "active_rules": active_rules,
            "feature_counts": feature_counts,
            "dialectal_markers": dialectal_markers,
            "has_dialectal_features": len(active_rules) > 0,
        }

    def annotate(self, text: str, dialect: Dialect) -> dict:
        """
        Annotate text with phonological metadata for downstream model consumption.
        
        Returns the original text alongside structured phonological annotations
        that can be used as additional features by the ASR/MT models.
        """
        analysis = self.analyze(text, dialect)
        
        # Build per-character annotations
        char_annotations = []
        for i, char in enumerate(text):
            anno = {"char": char, "index": i, "features": []}
            for rule in self.get_rules(dialect):
                if rule.source_pattern and char == rule.source_pattern:
                    anno["features"].append(rule.name)
            char_annotations.append(anno)

        return {
            "text": text,
            "dialect": dialect.value,
            "analysis": analysis,
            "char_annotations": char_annotations,
        }

    def detect_dialect(self, text: str) -> list[tuple[Dialect, float]]:
        """
        Heuristic dialect detection based on phonological feature density.
        
        Returns a ranked list of (dialect, confidence_score) tuples.
        This is a simple heuristic — not a trained classifier.
        """
        scores = []
        for dialect in Dialect:
            if dialect == Dialect.HINDI:
                continue
            analysis = self.analyze(text, dialect)
            if analysis["feature_counts"]:
                total_features = sum(analysis["feature_counts"].values())
                # Normalize by text length to get feature density
                density = total_features / max(len(text), 1)
                scores.append((dialect, density))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_hindi_cognate_hints(self, text: str, dialect: Dialect) -> list[dict]:
        """
        Generate hints about potential Hindi cognates for dialectal words.
        
        Useful for the MT model to understand Hindi-dialect correspondences.
        Returns mapping suggestions (not automatic replacements).
        """
        hints = []
        rules = self.get_rules(dialect)

        for rule in rules:
            if not rule.source_pattern or not rule.target_pattern:
                continue
            if rule.source_pattern in text:
                hints.append({
                    "rule": rule.name,
                    "dialectal_pattern": rule.source_pattern,
                    "hindi_equivalent": rule.target_pattern,
                    "context": rule.context,
                    "confidence": "high" if rule.priority >= 10 else "medium",
                })

        return hints

    @staticmethod
    def list_all_rules() -> dict[str, list[dict]]:
        """List all phonological rules across all dialects (for documentation)."""
        result = {}
        for dialect, rules in DIALECT_RULES.items():
            result[dialect.value] = [
                {
                    "name": r.name,
                    "source": r.source_pattern,
                    "target": r.target_pattern,
                    "context": r.context,
                    "priority": r.priority,
                }
                for r in rules
            ]
        return result
