"""Core domain types for face recognition."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BBox:
    """A scored face bounding box in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def wh(self) -> tuple[float, float]:
        """Return the box width and height."""
        return self.x2 - self.x1, self.y2 - self.y1


@dataclass(frozen=True, eq=False)
class Embedding:
    """An immutable, validated one-dimensional face embedding."""

    vector: np.ndarray[Any, np.dtype[Any]]
    normalized: bool = False

    def __post_init__(self) -> None:
        vector = np.array(self.vector, dtype=np.float32, order="C", copy=True)

        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"embedding must be 1-D non-empty, got shape {vector.shape}")

        if not np.isfinite(vector).all():
            raise ValueError("embedding contains NaN/Inf")

        if self.normalized:
            norm = float(np.linalg.norm(vector.astype(np.float64)))
            if not np.isclose(norm, 1.0, atol=1e-3):
                raise ValueError(f"normalized=True but ‖v‖={norm:.4g}")

        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)

    def normalize(self) -> "Embedding":
        """Return an L2-normalized embedding."""
        if self.normalized:
            return self

        norm = float(np.linalg.norm(self.vector.astype(np.float64)))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("cannot normalize an embedding with zero or non-finite norm")

        return Embedding(self.vector.astype(np.float64) / norm, normalized=True)

    def __array__(
        self,
        dtype: np.dtype[Any] | type[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Return the embedding as a NumPy array."""
        return np.array(self.vector, dtype=dtype, copy=copy)


@dataclass(eq=False)
class Face:
    """A detected face and the data produced while processing it."""

    bbox: BBox | None
    kps: np.ndarray[Any, np.dtype[Any]] | None
    aligned: np.ndarray[Any, np.dtype[Any]] | None
    embedding: Embedding | None
    image_index: int = 0
    face_index: int = 0

    def __post_init__(self) -> None:
        if self.kps is not None:
            self.kps = np.array(self.kps, copy=True)

        if self.aligned is not None:
            self.aligned = np.array(self.aligned, copy=True)


@dataclass(frozen=True)
class ComparisonResult:
    """Face-comparison metrics and the cosine-threshold decision."""

    cosine: float
    euclidean: float
    percentage: float
    is_match: bool
    threshold: float


@dataclass(frozen=True, eq=False)
class Match:
    """A scored face pair."""

    query: Face
    candidate: Face
    score: float
    percentage: float
    is_match: bool


def _default_match_threshold() -> float:
    from lvface.metrics import DEFAULT_THRESHOLDS

    return DEFAULT_THRESHOLDS["cosine"]


@dataclass(frozen=True, eq=False)
class MatchResult:
    """A face-similarity matrix and the pairs selected from it."""

    matrix: np.ndarray[Any, np.dtype[Any]]
    query_faces: list[Face]
    candidate_faces: list[Face]
    pairs: list[Match]
    threshold: float = field(default_factory=_default_match_threshold)

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=np.float64, order="C", copy=True)
        expected_shape = (len(self.query_faces), len(self.candidate_faces))

        if matrix.ndim != 2 or matrix.shape != expected_shape:
            raise ValueError(f"match matrix must have shape {expected_shape}, got {matrix.shape}")

        if not np.isfinite(matrix).all():
            raise ValueError("match matrix contains NaN/Inf")

        if not np.isfinite(self.threshold) or not -1.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be finite and between -1 and 1")

        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "query_faces", list(self.query_faces))
        object.__setattr__(self, "candidate_faces", list(self.candidate_faces))
        object.__setattr__(self, "pairs", list(self.pairs))

    def best_for(self, query_index: int) -> Match | None:
        """Return the highest-scoring selected match for one query face."""
        if not 0 <= query_index < len(self.query_faces):
            raise IndexError(f"query index out of range: {query_index}")

        query = self.query_faces[query_index]
        return max(
            (pair for pair in self.pairs if pair.query is query),
            key=lambda pair: pair.score,
            default=None,
        )

    def as_percentage_matrix(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Convert the similarity matrix to threshold-anchored display percentages."""
        from lvface.metrics import to_percentage

        percentages = np.fromiter(
            (to_percentage(float(score), threshold=self.threshold) for score in self.matrix.flat),
            dtype=np.float64,
            count=self.matrix.size,
        )
        return percentages.reshape(self.matrix.shape)
