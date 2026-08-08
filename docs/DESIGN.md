# Rajasthani Dialect AI — Linguistic Design Document

## Purpose

This document fills the critical gap identified in the architecture analysis:
there was no dialect-to-script mapping documentation explaining how localized
phonemes and grammar structures are programmatically mapped into standard
Devanagari by the preprocessing pipeline.

---

## Supported Dialects

| Dialect | Region | Lexical Similarity to Hindi | Data Availability |
|---|---|---|---|
| Marwari | Western Rajasthan | ~60% | Moderate (41+ hrs audio) |
| Mewari | Southern Rajasthan | ~55% | Low |
| Dhundhari | Eastern Rajasthan | ~65% | Low |
| Hadoti (Harauti) | Southeastern Rajasthan | ~60% | Very Low (0.34 hrs) |
| Mewati | Northeastern Rajasthan | ~50% | Very Low (0.35 hrs) |
| Bagri | Northern Rajasthan | ~55% (Gujarati influence) | Very Low (0.63 hrs) |

---

## Phonological Shift Mappings

### 1. Hindi /s/ → Marwari /h/ (Primary Shift)

The most documented and systematic sound change. Hindi sibilant /s/ (स)
becomes aspirate /h/ (ह) in Marwari across word-initial and medial positions.

| Hindi | Marwari | Meaning |
|---|---|---|
| सोना (sona) | होनो (hono) | gold |
| सात (saat) | हात (haat) | seven |
| सुंदर (sundar) | हुंदर (hundar) | beautiful |

**Implementation:** Encoded as a bidirectional `PhonologicalRule` in
`src/preprocessing/phonological_mapper.py` with priority=10.

### 2. Retroflex Lateral Flap ळ (U+0933)

**Critical preservation target.** ळ is heavily utilized in Marwari and Mewari
literature but is **entirely absent** from standard Hindi, which uses ल (U+0932)
or ड़ (U+095C) instead.

| Dialect Form | Hindi Equivalent | Note |
|---|---|---|
| वाळो (vaalo) | वालो (vaalo) | The ळ carries distinct phonetic information |
| काळो (kaalo) | कालो (kaalo) | Must never be normalized to ल |

**Implementation:** The `DevanagariNormalizer` explicitly verifies ळ survives
all normalization steps. The `_verify_retroflex_flap()` method audits and counts
preserved instances.

### 3. Okarant → Akarant Pluralization

Marwari singular nouns ending in okarant (ो) shift to akarant (ा) in plural,
diverging from standard Hindi conventions.

| Form | Example | Hindi Equivalent |
|---|---|---|
| Singular | घरो (gharo) | घर (ghar) |
| Plural | घरा (ghara) | घर (ghar) — Hindi doesn't change |

### 4. Nasalization Patterns

Marwari uses chandrabindu (ँ) more extensively for nasalization where Hindi
would use anusvara (ं).

### 5. Mewari Vowel Patterns

Mewari relies heavily on specific A and O vowel sounds during verb conjugations,
creating distinct prosodic patterns that must be captured by the TTS pitch model.

### 6. Bagri Gujarati Influence

Bagri exhibits strong phonetic influences from neighboring Gujarati.
Specific mappings require linguist validation on real data before encoding.

---

## Unicode Normalization Pipeline

### Execution Order (Non-negotiable)

```
1. Unicode NFC normalization
2. Nukta character decomposition (precomposed → base + modifier)
3. Vowel normalization
4. Retroflex lateral flap (ळ) verification
5. Whitespace normalization
6. [Optional] Non-Devanagari stripping
```

### Nukta Decomposition Map

This is the **single most critical** normalization step. Without it, the BPE
tokenizer treats precomposed and decomposed forms as different tokens.

| Precomposed | Decomposed | Name |
|---|---|---|
| क़ (U+0958) | क + ़ (U+0915 + U+093C) | Qa |
| ख़ (U+0959) | ख + ़ (U+0916 + U+093C) | Khha |
| ग़ (U+095A) | ग + ़ (U+0917 + U+093C) | Ghha |
| ज़ (U+095B) | ज + ़ (U+091C + U+093C) | Za |
| ड़ (U+095C) | ड + ़ (U+0921 + U+093C) | Dddha |
| ढ़ (U+095D) | ढ + ़ (U+0922 + U+093C) | Rha |
| फ़ (U+095E) | फ + ़ (U+092B + U+093C) | Fa |
| य़ (U+095F) | य + ़ (U+092F + U+093C) | Yya |

**Note:** NFC normalization (Step 1) already decomposes these characters in
Python's `unicodedata` module. Step 2 acts as a redundant safety net using
`str.translate()` to catch any edge cases NFC misses.

---

## Sentence Structure

All six dialects follow Subject-Object-Verb (SOV) word order, consistent with
Hindi. This structural similarity is a key enabler for cross-lingual transfer
learning.

---

## Known Gaps Requiring Linguist Validation

1. **Hadoti rules:** No phonological rules have been encoded yet.
2. **Mewati rules:** No phonological rules have been encoded yet.
3. **Bagri-Gujarati specific mappings:** Requires field linguist input.
4. **Code-switching boundaries:** No formal rules for detecting Hindi↔dialect
   switches within a single sentence.
