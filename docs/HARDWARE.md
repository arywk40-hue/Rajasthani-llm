# Rajasthani Dialect AI — Edge Hardware Specification

## Purpose

This document fills the critical gap identified in the architecture analysis:
there was no hardware specification document for the Suno Sutra edge device
detailing the physical-to-digital bridge constraints.

---

## Suno Sutra Device Profile

| Parameter | Specification | Notes |
|---|---|---|
| **Device Type** | Handheld edge AI device | Voice-first, conversational AI |
| **Connectivity** | Fully offline capable | Zero cloud dependency |
| **Target Users** | Rural demographics | Low/no broadband access |
| **Runtime** | C++ embedded runtime | Not Python — requires compiled models |

---

## Hardware Constraints (Inferred)

> **⚠ Important:** Exact SoC architecture, RAM specs, and microphone array
> geometry are NOT publicly documented. The values below are inferred from
> the architecture documentation and must be validated with the hardware team.

| Constraint | Value | Source |
|---|---|---|
| **Max Volatile Memory** | 900 MB | Architecture doc: "900-megabyte volatile memory footprint" |
| **Quantization Format** | FP16 / INT8 | Architecture doc: "FP16 mathematical quantization" |
| **Inference Runtime** | CTranslate2 / ONNX Runtime | Architecture doc: deployment layer |
| **Audio Input** | 16 kHz sampling rate | ASR pipeline requirement |
| **Audio Output** | 22050 Hz sampling rate | TTS vocoder (HiFi-GAN V1) |

---

## Model Memory Budget

The combined memory footprint of all three S2ST models must fit within 900MB.

| Component | Model | Parameters | Est. FP16 Size | Est. INT8 Size |
|---|---|---|---|---|
| ASR | FastConformer | 430M | ~860 MB | ~430 MB |
| MT | IndicTrans2 | 1B (distilled 200M for edge) | ~400 MB | ~200 MB |
| TTS (Acoustic) | FastPitch | ~20M | ~40 MB | ~20 MB |
| TTS (Vocoder) | HiFi-GAN V1 | ~14M | ~28 MB | ~14 MB |
| **Total (Full)** | | | **~1328 MB** | **~664 MB** |

> **Key Insight:** The full 1B IndicTrans2 model does NOT fit on edge.
> The 200M distilled variant (requiring 950MB VRAM per the server docs,
> but ~200MB INT8 on disk) must be used for edge deployment.
> INT8 quantization of all components fits within the 900MB ceiling.

---

## Deployment Pipeline

```
PyTorch Model (Training)
    │
    ├── INT8 Dynamic Quantization (torch.quantization.quantize_dynamic)
    │       └── .pt checkpoint
    │
    ├── ONNX Export (torch.onnx.export, opset 14+)
    │       └── .onnx model → onnxruntime inference
    │
    └── CTranslate2 Conversion (for NMT specifically)
            └── model.bin → ctranslate2 inference (optimized Transformer)
```

---

## Thermal and Battery Considerations

Running a full three-stage S2ST cascade on a handheld device presents:

1. **Thermal throttling:** Continuous inference may trigger thermal limits.
   Mitigation: batch requests, idle cooldown periods.
2. **Battery drain:** GPU/NPU-intensive inference depletes battery rapidly.
   Mitigation: INT8 CPU inference preferred over FP16 GPU.
3. **Memory fragmentation:** Long-running sessions may fragment the heap.
   Mitigation: model loading at boot, no dynamic allocation during inference.

---

## Validation Checklist

Before shipping an edge build:

- [ ] All quantized models fit within 900MB combined
- [ ] ONNX models pass `onnx.checker.check_model()` validation
- [ ] CTranslate2 NMT model produces valid UTF-8 Devanagari output
- [ ] End-to-end latency < 2 seconds for a 5-second audio input
- [ ] Device operates for ≥ 4 hours continuous use on full battery
- [ ] No network calls made during offline inference
