"""
Model Quantizer for Edge Deployment

Implements post-training quantization to INT8 and FP16 formats to optimize
the models for mobile and edge devices.

Reference: architecture-documenattion.md (Edge Deployment Architecture)
Reference: implementation.md Section 3 (Edge deployment — INT8/FP16 quantization)
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


class QuantizationType(Enum):
    """Supported quantization formats for edge deployment."""
    INT8 = "int8"
    FP16 = "fp16"


class ModelQuantizer:
    """
    Quantizes PyTorch models for edge deployment.

    Post-training quantization reduces the Whisper ASR model and the 
    IndicTrans2 MT model to deployable sizes without retraining.

    Supports:
    - INT8 dynamic quantization (CPU inference, smallest footprint)
    - FP16 half-precision (GPU/NPU inference, balanced accuracy-speed)
    """

    MAX_EDGE_MEMORY_MB = 1024  # Standard 1GB edge memory budget

    def __init__(self, output_dir: str | Path = "models/quantized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelQuantizer initialized. Output: {self.output_dir}")

    def quantize_int8(self, model: nn.Module, model_name: str) -> Path:
        """
        Apply INT8 dynamic quantization to a PyTorch model.

        Dynamic quantization converts weights to INT8 at save time and
        quantizes activations dynamically at inference time. Best suited
        for CPU-bound inference on the Suno Sutra device.
        """
        logger.info(f"Quantizing {model_name} to INT8 (dynamic)...")

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d},  # Layers to quantize
            dtype=torch.qint8,
        )

        output_path = self.output_dir / f"{model_name}_int8.pt"
        torch.save(quantized_model.state_dict(), output_path)

        original_size = self._get_model_size_mb(model)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)

        logger.info(
            f"INT8 quantization complete: {model_name} | "
            f"Original: {original_size:.1f}MB → Quantized: {quantized_size:.1f}MB "
            f"({(1 - quantized_size / max(original_size, 0.1)) * 100:.1f}% reduction)"
        )

        if quantized_size > self.MAX_EDGE_MEMORY_MB:
            logger.warning(
                f"⚠ Quantized model ({quantized_size:.1f}MB) exceeds "
                f"the edge memory limit ({self.MAX_EDGE_MEMORY_MB}MB). "
                f"Further pruning or distillation required."
            )

        return output_path

    def quantize_fp16(self, model: nn.Module, model_name: str) -> Path:
        """
        Convert model weights to FP16 half-precision.

        FP16 provides a 2x memory reduction with minimal accuracy loss.
        Suitable for devices with GPU/NPU hardware that natively supports
        half-precision arithmetic.
        """
        logger.info(f"Converting {model_name} to FP16...")

        model_fp16 = model.half()
        output_path = self.output_dir / f"{model_name}_fp16.pt"
        torch.save(model_fp16.state_dict(), output_path)

        original_size = self._get_model_size_mb(model)
        fp16_size = os.path.getsize(output_path) / (1024 * 1024)

        logger.info(
            f"FP16 conversion complete: {model_name} | "
            f"Original: {original_size:.1f}MB → FP16: {fp16_size:.1f}MB"
        )

        return output_path

    def validate_edge_budget(self, *model_paths: Path) -> bool:
        """
        Validates that the combined size of all quantized models fits
        within the target edge memory budget.
        """
        total_mb = sum(
            os.path.getsize(p) / (1024 * 1024) for p in model_paths if p.exists()
        )
        fits = total_mb <= self.MAX_EDGE_MEMORY_MB
        status = "✓ PASS" if fits else "✗ FAIL"
        logger.info(
            f"Edge budget check: {status} | "
            f"Total: {total_mb:.1f}MB / {self.MAX_EDGE_MEMORY_MB}MB"
        )
        return fits

    @staticmethod
    def _get_model_size_mb(model: nn.Module) -> float:
        """Estimate model size in MB from parameter count."""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
