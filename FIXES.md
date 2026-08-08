# FIXES.md — Rajasthani Dialect AI

## Critical Fixes Applied

### 1. Config Mismatch: ASR Architecture
**File:** `config/asr.yaml`
**Issue:** Config specified FastConformer (NeMo) but `src/asr/model.py` implemented Whisper/HuggingFace
**Fix:** Rewrote config to match Whisper architecture with `hf_model_name`, `freeze_encoder`, `gradient_checkpointing` fields

### 2. Invalid Model Registry Entry
**File:** `src/asr/model.py:42`
**Issue:** `"saaras-v1": "sarvamai/saaras-v1"` — Sarvam Saaras is API-only, not on HuggingFace
**Fix:** Commented out with explanatory note

### 3. Absolute Import Breaking Without `pip install -e .`
**File:** `src/asr/model.py:398`
**Issue:** `from src.evaluation.metrics import compute_cer, compute_wer` fails when package not installed
**Fix:** Changed to relative import: `from ..evaluation.metrics import compute_cer, compute_wer`

### 4. Missing MT Inference Entry Point
**Files Created:** `src/mt/translate.py`
**Issue:** No `IndicTrans2Translator` class for API inference — API couldn't call MT
**Fix:** Created production wrapper mirroring `WhisperASR` pattern with `translate()` method

### 5. Missing TTS Inference Entry Point
**Files Created:** `src/tts/synthesize.py`
**Issue:** No `IndicTTSSynthesizer` class for API inference — API couldn't call TTS
**Fix:** Created production wrapper for AI4Bharat Indic-TTS (FastPitch + HiFi-GAN)

### 6. API Endpoints Returning Hardcoded Placeholders
**File:** `src/api/routes/translate.py`
**Issue:** `/translate`, `/asr`, `/tts` returned static strings instead of calling models
**Fix:** Wired endpoints to `request.app.state.{mt_model,asr_model,tts_model}` with proper error handling

### 7. Model Loading in API Lifespan
**File:** `src/api/app.py`
**Issue:** Lifespan tried to load non-existent `IndicTTS` from `fastpitch.py`
**Fix:** Updated to load `WhisperASR`, `IndicTrans2MT`, `IndicTTSSynthesizer`

### 8. Broken Module Exports
**Files:** `src/asr/__init__.py`, `src/tts/__init__.py`, `src/tts/trainer.py`
**Issue:** Exported non-existent classes (`FastConformerASR`, `FastPitchAcoustic`, `HiFiGANVocoder`)
**Fix:** Updated exports to match actual implementations

---

## Minor Fixes Applied

### 9. TTS Synthesize Method Signature
**File:** `src/api/routes/translate.py:163`
**Issue:** Called `model.synthesize(text, dialect=...)` but method doesn't accept dialect param
**Fix:** Removed `dialect` kwarg

---

## Tests Status

```bash
pytest tests/test_normalizer.py -v
# 47 passed in 0.04s
```

---

## Remaining Work (Not Yet Implemented)

### A. Real Training Loops
| Script | Status | Needed |
|--------|--------|--------|
| `scripts/train_asr.py` | Dummy tensors | Real CTC loss, torchaudio features, HF Trainer integration |
| `scripts/train_mt.py` | Dummy tensors | Experience replay sampler, model souping callback |
| `scripts/train_tts.py` | Dummy tensors | FastPitch/HiFi-GAN loss functions |

### B. Data Ingestion (Audio Files)
| Script | Status | Needed |
|--------|--------|--------|
| `src/data/fetch_datasets.py` | Metadata only | `fetch_vaani_with_audio()` downloads audio arrays to `.wav` files |

### C. Benchmark Runner
| Need | Description |
|------|-------------|
| Zero-shot baseline | Run IndicWhisper/Sarvam/Whisper-large-v3 on VAANI test split → WER/CER table |

### D. Environment Fix
| Issue | Fix |
|-------|-----|
| `torchaudio` version mismatch | `pip install --upgrade torchaudio --index-url https://download.pytorch.org/whl/cpu` |

---

## Verification Commands

```bash
# All imports work
python -c "
from src.mt.translate import IndicTrans2Translator
from src.tts.synthesize import IndicTTSSynthesizer
from src.asr.model import WhisperASR
from src.api.app import app
print('OK')
"

# Tests pass
pytest tests/test_normalizer.py -v

# API starts (needs torchaudio fix for ASR)
uvicorn src.api.app:app --reload
```