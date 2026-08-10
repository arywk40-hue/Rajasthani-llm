"""
Dialect-Aware Translation Pipeline

Implements clean abstractions for dialect-specific machine translation.
Routes inputs through dialect-specific normalizers and maps to standard Hindi
cognates before executing the baseline translation model.
"""

from typing import Optional, List, Dict
from src.mt.model import IndicTrans2MT
from src.preprocessing.phonological_mapper import PhonologicalMapper, Dialect
from src.preprocessing.normalizer import DevanagariNormalizer
from loguru import logger


class DialectConfig:
    """Explicit configuration of dialect properties and model status."""
    
    def __init__(self, dialect_name: str, is_finetuned: bool = False):
        self.dialect_name = dialect_name.lower().strip()
        self.is_finetuned = is_finetuned
        self.status = "DIALECT-FINETUNED" if is_finetuned else "BASELINE - NOT DIALECT-FINETUNED"


class DialectNormalizer:
    """Normalizes dialect text by mapping specific vocabulary and phonological tokens to Hindi."""
    
    def __init__(self):
        self.deva_normalizer = DevanagariNormalizer()
        
        # Dialect lexical mappings for cognate generation
        self.lexicon: Dict[str, Dict[str, str]] = {
            "marwari": {
                "टाबर": "बच्चे",
                "काका": "चाचा",
                "अठै": "यहाँ",
                "कठै": "कहाँ",
                "चोखो": "ठीक",
                "म्हारो": "मेरा",
                "थारो": "तुम्हारा",
                "कांई": "क्या",
                "होनो": "सोना",
                "कोनी": "नहीं",
            },
            "mewari": {
                "अणी": "इस",
                "कइ": "क्या",
                "म्हारो": "मेरा",
                "कठै": "कहाँ",
            },
            "dhundhari": {
                "छोरा": "लड़का",
                "छोरी": "लड़की",
                "थाको": "तुम्हारा",
                "कांई": "क्या",
                "कठै": "कहाँ",
                "छै": "है",
                "रयो": "रहा",
            },
            "hadoti": {
                "काय": "क्या",
                "छै": "है",
                "अठै": "यहाँ",
            },
            "mewati": {
                "रह्यो": "रहा",
                "कहाँ": "कहाँ",
            },
            "bagri": {
                "किन्नै": "किस तरफ",
                "जावैगा": "जाओगे",
                "थांरो": "तुम्हारा",
                "साग": "सब्जी",
            }
        }

    def normalize_to_hindi(self, text: str, dialect_name: str) -> str:
        """Map dialectal terms and characters to standard Hindi equivalents."""
        dialect_lower = dialect_name.lower().strip()
        
        # 1. Unicode NFC & Devanagari normalization
        text = self.deva_normalizer.normalize(text)
        
        # 2. Lexical word mapping
        if dialect_lower in self.lexicon:
            words = text.split()
            mapped_words = []
            for w in words:
                # Strip punctuation for dictionary lookup
                clean_w = w.strip("।,!?")
                if clean_w in self.lexicon[dialect_lower]:
                    replacement = self.lexicon[dialect_lower][clean_w]
                    w = w.replace(clean_w, replacement)
                mapped_words.append(w)
            text = " ".join(mapped_words)
            
        # 3. Phonological conversions (e.g., retroflex lateral flap ळ -> standard ल)
        text = text.replace("ळ", "ल")
            
        return text


class TranslationPipeline:
    """Manages the explicit dialect-aware translation pipeline flow."""
    
    def __init__(self, mt_model: Optional[IndicTrans2MT] = None, model_name: Optional[str] = None):
        self.mt_model = mt_model or IndicTrans2MT(model_name=model_name)
        self.normalizer = DialectNormalizer()
        logger.info("TranslationPipeline initialized with DialectNormalizer")

    def translate_dialect(self, text: str, dialect: str, target_lang: str) -> dict:
        """
        Translates dialectal text using normalizers followed by standard MT translation.
        """
        dialect_lower = dialect.lower().strip()
        config = DialectConfig(dialect_lower)
        
        # Normalize dialect to standard Hindi cognates
        hindi_cognate = self.normalizer.normalize_to_hindi(text, dialect_lower)
        
        target_lang_lower = target_lang.lower().strip()
        
        # If target language is Hindi, return the normalized cognate directly
        if target_lang_lower in ("hindi", "hi", "hin_deva"):
            translated = hindi_cognate
        else:
            # Otherwise translate standard Hindi to English/Target
            translations = self.mt_model.translate(
                [hindi_cognate],
                src_lang="hindi",
                tgt_lang=target_lang_lower
            )
            translated = translations[0] if translations else ""
            
        return {
            "dialect": dialect_lower,
            "original_text": text,
            "hindi_cognate": hindi_cognate,
            "translated_text": translated,
            "target_lang": target_lang_lower,
            "status": config.status
        }
