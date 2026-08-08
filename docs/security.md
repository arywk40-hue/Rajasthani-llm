# Security Specification

## Security Principles

The security architecture of the **Rajasthani Dialect AI** platform ensures confidentiality, integrity, and availability across all model training, API endpoints, and data storage workflows.

---

## 1. API & Access Security

- **Transport Security:** All web endpoints (`src/api/`) communicate exclusively over HTTPS/TLS 1.3.
- **Authentication & Rate Limiting:** External API calls to model endpoints are rate-limited and require Bearer API Token authentication.
- **Input Validation:** Audio inputs (WAV, MP3, FLAC) are subject to strict payload length verification (max 10MB per request) and format header checks to prevent buffer overflow attacks.

---

## 2. Model & Infrastructure Integrity

- **Model Weight Hash Verification:** Pretrained weights downloaded from HuggingFace (`vasista22/whisper-hindi-large-v2`, `AI4Bharat/IndicTrans2`) are validated against SHA-256 checksums before loading into memory.
- **Environment Isolation:** Model execution runs within sandboxed Docker containers (`Dockerfile`) with non-root user permissions.

---

## 3. Vulnerability & Dependency Management

- **Dependency Auditing:** Python dependencies in `requirements.txt` and `pyproject.toml` are scanned for known CVEs using automated security scanners.
- **Secrets Management:** API keys (e.g., Bhashini API tokens) are loaded strictly through environment variables and never hardcoded in source repositories.
