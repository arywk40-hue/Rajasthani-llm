# Rajasthani Dialect AI — API Specification

## Base URL

```
https://api.bhashini.gov.in/rajasthani/v1
```

## Authentication

All endpoints require an `X-API-Key` header issued via the Bhashini NHLT portal.

```
X-API-Key: <your-api-key>
```

---

## Endpoints

### `GET /api/v1/health`

Health check. No authentication required.

**Response 200:**
```json
{
  "status": "healthy",
  "service": "rajasthani-dialect-ai",
  "version": "0.1.0"
}
```

---

### `POST /api/v1/translate`

Translate text between Rajasthani dialects and Hindi/English.

**Request:**
```json
{
  "text": "होनो कहते हैं",
  "src_lang": "mar_Deva",
  "tgt_lang": "hin_Deva"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✓ | Source text (1–2048 chars, Devanagari) |
| `src_lang` | string | ✓ | FLORES-200 source language code |
| `tgt_lang` | string | ✓ | FLORES-200 target language code |

**Response 200:**
```json
{
  "translated_text": "सोना कहते हैं",
  "src_lang": "mar_Deva",
  "tgt_lang": "hin_Deva",
  "model_version": "0.1.0"
}
```

**Language Codes:**

| Dialect | Code |
|---|---|
| Marwari | `mar_Deva` |
| Mewari | `mew_Deva` |
| Dhundhari | `dhu_Deva` |
| Hadoti | `had_Deva` |
| Mewati | `mwt_Deva` |
| Bagri | `bgr_Deva` |
| Hindi | `hin_Deva` |
| English | `eng_Latn` |

---

### `POST /api/v1/asr`

Transcribe audio to Devanagari text via FastConformer ASR.

**Request:**
```json
{
  "audio_base64": "<base64-encoded-16kHz-WAV>",
  "dialect": "marwari"
}
```

**Response 200:**
```json
{
  "transcribed_text": "मारवाड़ी में होनो कहते हैं",
  "dialect": "marwari",
  "confidence": 0.92
}
```

---

### `POST /api/v1/tts`

Synthesize speech from text using FastPitch + HiFi-GAN.

**Request:**
```json
{
  "text": "मारवाड़ी में होनो कहते हैं",
  "dialect": "marwari"
}
```

**Response 200:**
```json
{
  "audio_base64": "<base64-encoded-WAV>",
  "sample_rate": 22050,
  "dialect": "marwari"
}
```

---

## Error Responses

| Status | Meaning |
|---|---|
| `401` | Invalid or missing `X-API-Key` |
| `422` | Validation error (bad request body) |
| `500` | Internal server error |

## Compliance Headers

Every response includes DPDP Act compliance headers:

| Header | Value | Description |
|---|---|---|
| `X-Request-ID` | UUID | Unique audit trail identifier |
| `X-Data-Localization` | `IN` | India-first data sovereignty |
| `X-Log-Retention-Days` | `30` | Maximum log retention window |
| `X-DPDP-Compliant` | `true` | DPDP Act 2023 compliance flag |
| `X-Powered-By` | `BHASHINI` | Mandatory attribution |

## Rate Limits

Per Bhashini NHLT gateway policy. Contact MeitY for enterprise tier limits.
