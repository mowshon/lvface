"""Modern face-embedding framework."""

from . import metrics
from .detect import FaceDetector, InsightFaceDetector
from .embed import FaceEmbedder, LVFaceOnnxEmbedder
from .errors import AlignmentError, NoFaceError
from .hub import resolve_weights
from .io import load_image
from .recognizer import FaceRecognizer
from .registry import DEFAULT_MODEL, MODELS, Model, resolve_model_path
from .types import BBox, ComparisonResult, Embedding, Face, Match, MatchResult

__version__ = "0.1.0"

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
