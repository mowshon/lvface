"""Modern face-embedding framework."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.2.0"

__all__ = [
    "AlignmentError",
    "BBox",
    "ComparisonResult",
    "DEFAULT_MODEL",
    "Embedding",
    "Face",
    "FaceDetector",
    "FaceEmbedder",
    "FaceRecognizer",
    "InsightFaceDetector",
    "LVFaceOnnxEmbedder",
    "MODELS",
    "Match",
    "MatchResult",
    "Model",
    "NoFaceError",
    "__version__",
    "load_image",
    "metrics",
    "resolve_model_path",
    "resolve_weights",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the public API without importing optional runtimes."""
    if name == "metrics":
        return importlib.import_module(".metrics", __name__)

    if name in {"AlignmentError", "NoFaceError"}:
        from .errors import AlignmentError, NoFaceError

        return {"AlignmentError": AlignmentError, "NoFaceError": NoFaceError}[name]

    if name in {"BBox", "ComparisonResult", "Embedding", "Face", "Match", "MatchResult"}:
        from .types import BBox, ComparisonResult, Embedding, Face, Match, MatchResult

        return {
            "BBox": BBox,
            "ComparisonResult": ComparisonResult,
            "Embedding": Embedding,
            "Face": Face,
            "Match": Match,
            "MatchResult": MatchResult,
        }[name]

    if name in {"DEFAULT_MODEL", "MODELS", "Model", "resolve_model_path"}:
        from .registry import DEFAULT_MODEL, MODELS, Model, resolve_model_path

        return {
            "DEFAULT_MODEL": DEFAULT_MODEL,
            "MODELS": MODELS,
            "Model": Model,
            "resolve_model_path": resolve_model_path,
        }[name]

    if name == "load_image":
        from .io import load_image

        return load_image

    if name == "resolve_weights":
        from .hub import resolve_weights

        return resolve_weights

    if name in {"FaceDetector", "InsightFaceDetector"}:
        from .detect import FaceDetector, InsightFaceDetector

        return {
            "FaceDetector": FaceDetector,
            "InsightFaceDetector": InsightFaceDetector,
        }[name]

    if name in {"FaceEmbedder", "LVFaceOnnxEmbedder"}:
        from .embed import FaceEmbedder, LVFaceOnnxEmbedder

        return {
            "FaceEmbedder": FaceEmbedder,
            "LVFaceOnnxEmbedder": LVFaceOnnxEmbedder,
        }[name]

    if name == "FaceRecognizer":
        from .recognizer import FaceRecognizer

        return FaceRecognizer

    raise AttributeError(f"module 'lvface' has no attribute {name!r}")
