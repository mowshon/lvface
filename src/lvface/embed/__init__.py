"""Face embedding backends."""

from .base import FaceEmbedder
from .onnx import LVFaceOnnxEmbedder

__all__ = ["FaceEmbedder", "LVFaceOnnxEmbedder"]
