"""
Edge Deployment Module

Provides model quantization (INT8/FP16) and ONNX/CTranslate2 export
utilities for deploying to the Suno Sutra handheld edge AI device.
"""

from src.edge.quantizer import ModelQuantizer
from src.edge.onnx_exporter import ONNXExporter

__all__ = ["ModelQuantizer", "ONNXExporter"]
