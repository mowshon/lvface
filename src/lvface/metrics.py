"""Similarity and distance metrics for face embeddings.

For L2-normalized vectors, squared Euclidean distance equals ``2 - 2 * cosine``.
Cosine is the sole decision metric; distances are diagnostic outputs.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

VectorLike = npt.ArrayLike
MatrixLike = npt.ArrayLike
DistanceMetric = Literal["euclidean", "euclidean_normalized"]
PercentageMethod = Literal["sigmoid", "linear"]


@dataclass(frozen=True)
class MetricSpec:
    """Metadata describing a comparison metric."""

    name: str
    higher_is_better: bool
    decision: bool


METRICS = {
    "cosine": MetricSpec("cosine", higher_is_better=True, decision=True),
    "euclidean": MetricSpec("euclidean", higher_is_better=False, decision=False),
    "euclidean_normalized": MetricSpec(
        "euclidean_normalized",
        higher_is_better=False,
        decision=False,
    ),
}

# Provisional ArcFace-style starting point; calibrate it for each model and deployment.
DEFAULT_THRESHOLDS = {"cosine": 0.35}


def _vectors(
    a: VectorLike,
    b: VectorLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate and convert two vector-like inputs.

    Args:
        a: First one-dimensional vector.
        b: Second one-dimensional vector.

    Returns:
        Two matching finite float64 vectors.
    """
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)

    if left.ndim != 1 or right.ndim != 1:
        raise ValueError(f"expected two 1-D vectors, got shapes {left.shape} and {right.shape}")

    if left.size == 0 or right.size == 0:
        raise ValueError("vectors must be non-empty")

    if left.shape != right.shape:
        raise ValueError(f"vector shapes must match, got {left.shape} and {right.shape}")

    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("vectors must contain only finite values")

    return left, right


def _matrices(
    a: MatrixLike,
    b: MatrixLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate and convert two matrix-like inputs.

    Args:
        a: Matrix whose rows are feature vectors.
        b: Matrix whose rows are feature vectors.

    Returns:
        Two finite float64 matrices with matching feature dimensions.
    """
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"expected two 2-D matrices, got shapes {left.shape} and {right.shape}")

    if left.shape[0] == 0 or right.shape[0] == 0 or left.shape[1] == 0 or right.shape[1] == 0:
        raise ValueError("matrices must have non-empty rows and feature dimensions")

    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"matrix feature dimensions must match, got {left.shape[1]} and {right.shape[1]}"
        )

    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("matrices must contain only finite values")

    return left, right


def _unit_vector(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """L2-normalize a finite, nonzero vector.

    Args:
        vector: Vector to normalize.

    Returns:
        The normalized vector.
    """
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot normalize a vector with zero or non-finite norm")

    return vector / norm


def _unit_rows(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """L2-normalize every row of a matrix.

    Args:
        matrix: Matrix containing one vector per row.

    Returns:
        A float64 matrix with unit-length rows.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise ValueError("cannot normalize rows with zero or non-finite norm")

    return np.asarray(matrix / norms, dtype=np.float64)


def cosine(a: VectorLike, b: VectorLike) -> float:
    """Return cosine similarity for two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in the range [-1, 1].
    """
    left, right = _vectors(a, b)
    return float(np.clip(np.dot(_unit_vector(left), _unit_vector(right)), -1.0, 1.0))


def euclidean(a: VectorLike, b: VectorLike) -> float:
    """Return Euclidean distance for two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Euclidean distance between the vectors.
    """
    left, right = _vectors(a, b)
    return float(np.linalg.norm(left - right))


def euclidean_normalized(a: VectorLike, b: VectorLike) -> float:
    """Return distance after independently L2-normalizing both vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Euclidean distance between the normalized vectors.
    """
    left, right = _vectors(a, b)
    return float(np.linalg.norm(_unit_vector(left) - _unit_vector(right)))


def dot(a: VectorLike, b: VectorLike) -> float:
    """Return the inner product of two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Dot product of the vectors.
    """
    left, right = _vectors(a, b)
    return float(np.dot(left, right))


def pairwise_cosine(a: MatrixLike, b: MatrixLike) -> npt.NDArray[np.float64]:
    """Compute cosine similarity between every pair of rows.

    Args:
        a: Matrix containing N feature vectors.
        b: Matrix containing M feature vectors.

    Returns:
        An ``(N, M)`` cosine-similarity matrix.
    """
    left, right = _matrices(a, b)
    return np.clip(_unit_rows(left) @ _unit_rows(right).T, -1.0, 1.0)


def pairwise_distance(
    a: MatrixLike,
    b: MatrixLike,
    metric: DistanceMetric = "euclidean",
) -> npt.NDArray[np.float64]:
    """Compute distance between every pair of rows.

    Args:
        a: Matrix containing N feature vectors.
        b: Matrix containing M feature vectors.
        metric: Distance metric to apply.

    Returns:
        An ``(N, M)`` distance matrix.
    """
    left, right = _matrices(a, b)
    if metric == "euclidean_normalized":
        left = _unit_rows(left)
        right = _unit_rows(right)
    elif metric != "euclidean":
        raise ValueError(f"unsupported distance metric: {metric!r}")

    # This form avoids allocating an (N, M, D) difference array.
    squared = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.asarray(np.sqrt(np.maximum(squared, 0.0)), dtype=np.float64)


def to_percentage(
    score: float,
    *,
    threshold: float | None = None,
    sharpness: float = 12.0,
    method: PercentageMethod = "sigmoid",
) -> float:
    """Convert cosine similarity to a display score, not a probability.

    The default sigmoid is centered at the cosine decision threshold. The linear
    method maps the full cosine range directly onto 0 through 100.

    Args:
        score: Cosine similarity to convert.
        threshold: Similarity mapped to 50 by the sigmoid method.
        sharpness: Steepness of the sigmoid curve.
        method: Conversion method, either ``"sigmoid"`` or ``"linear"``.

    Returns:
        A display percentage in the range [0, 100].
    """
    if not np.isfinite(score):
        raise ValueError("score must be finite")

    if method == "linear":
        return float(np.clip((score + 1.0) * 50.0, 0.0, 100.0))

    if method != "sigmoid":
        raise ValueError(f"unsupported percentage method: {method!r}")

    boundary = DEFAULT_THRESHOLDS["cosine"] if threshold is None else threshold
    if not np.isfinite(boundary):
        raise ValueError("threshold must be finite")

    if not np.isfinite(sharpness) or sharpness <= 0.0:
        raise ValueError("sharpness must be finite and greater than zero")

    # Split the sigmoid to keep large exponents numerically stable.
    exponent = sharpness * (score - boundary)
    if exponent >= 0.0:
        percentage = 100.0 / (1.0 + np.exp(-exponent))
    else:
        exp_value = np.exp(exponent)
        percentage = 100.0 * exp_value / (1.0 + exp_value)
    return float(np.clip(percentage, 0.0, 100.0))
