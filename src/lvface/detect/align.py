"""Five-point ArcFace alignment without OpenCV."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

from lvface.errors import AlignmentError

ARCFACE_DST: npt.NDArray[np.float32] = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
ARCFACE_DST.setflags(write=False)


def _landmarks_array(kps: object) -> npt.NDArray[np.float64]:
    """Validate five-point landmarks as a float64 array.

    Args:
        kps: Array-like landmark coordinates.

    Returns:
        A finite, nondegenerate array with shape ``(5, 2)``.
    """
    try:
        landmarks = np.asarray(kps, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentError("landmarks must be a numeric array with shape (5, 2)") from error

    if landmarks.shape != (5, 2):
        raise AlignmentError(f"need exactly 5 landmarks with shape (5, 2), got {landmarks.shape}")

    if not np.isfinite(landmarks).all():
        raise AlignmentError("landmarks must contain only finite coordinates")

    if np.linalg.matrix_rank(landmarks - landmarks.mean(axis=0)) < 2:
        raise AlignmentError("degenerate landmark geometry; cannot align")

    return landmarks


def _umeyama(
    source: npt.NDArray[np.float64],
    destination: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | None:
    """Estimate a 2D similarity transform with Umeyama's method.

    Args:
        source: Source points with shape ``(N, 2)``.
        destination: Destination points with shape ``(N, 2)``.

    Returns:
        A ``(2, 3)`` affine matrix, or ``None`` for degenerate points.
    """
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    source_variance = np.sum(source_centered**2) / source.shape[0]
    if source_variance <= np.finfo(np.float64).eps:
        return None

    covariance = destination_centered.T @ source_centered / source.shape[0]
    u, singular_values, vh = np.linalg.svd(covariance)
    signs = np.ones(source.shape[1], dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        signs[-1] = -1.0

    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        return None

    if rank == source.shape[1] - 1:
        if np.linalg.det(u) * np.linalg.det(vh) > 0:
            rotation = u @ vh
        else:
            final_sign = signs[-1]
            signs[-1] = -1.0
            rotation = u @ np.diag(signs) @ vh
            signs[-1] = final_sign
    else:
        rotation = u @ np.diag(signs) @ vh

    scale = float(singular_values @ signs) / source_variance
    matrix = np.empty((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = destination_mean - scale * rotation @ source_mean
    return matrix


def _invert_affine(matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Invert a two-dimensional affine transform.

    Args:
        matrix: Affine matrix with shape ``(2, 3)``.

    Returns:
        The inverse affine matrix.
    """
    linear = np.asarray(matrix[:, :2], dtype=np.float64)
    translation = np.asarray(matrix[:, 2], dtype=np.float64)
    try:
        inverse_linear = np.linalg.inv(linear)
    except np.linalg.LinAlgError as error:
        raise AlignmentError("alignment transform is not invertible") from error

    inverse = np.empty((2, 3), dtype=np.float64)
    inverse[:, :2] = inverse_linear
    with np.errstate(invalid="ignore"):
        inverse[:, 2] = -inverse_linear @ translation

    if not np.isfinite(inverse).all():
        raise AlignmentError("alignment transform is not finite")

    return inverse


def estimate_norm(kps: object, size: int = 112) -> npt.NDArray[np.float32]:
    """Estimate the transform from landmarks to the ArcFace template.

    Args:
        kps: Five facial landmarks with shape ``(5, 2)``.
        size: Output crop size; LVFace supports only 112.

    Returns:
        A float32 affine matrix with shape ``(2, 3)``.
    """
    if size != 112:
        raise AlignmentError("only size=112 is supported for LVFace alignment")

    landmarks = _landmarks_array(kps)
    matrix = _umeyama(landmarks, ARCFACE_DST.astype(np.float64))
    if matrix is None or not np.isfinite(matrix).all():
        raise AlignmentError("degenerate landmark geometry; cannot align")

    return matrix.astype(np.float32)


def _validate_image(image: object) -> npt.NDArray[np.uint8]:
    """Validate an RGB uint8 image array.

    Args:
        image: Object expected to be a non-empty RGB image.

    Returns:
        The validated image without copying it.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must have non-zero shape (H, W, 3), got {image.shape}")

    if image.dtype != np.uint8:
        raise ValueError(f"image must have dtype uint8, got {image.dtype}")

    return image


def norm_crop(
    image: np.ndarray,
    kps: object,
    size: int = 112,
) -> npt.NDArray[np.uint8]:
    """Align an RGB image to an ArcFace crop using Pillow.

    Args:
        image: Source RGB uint8 image.
        kps: Five facial landmarks with shape ``(5, 2)``.
        size: Output width and height; LVFace supports only 112.

    Returns:
        An owned aligned RGB uint8 crop.
    """
    source = _validate_image(image)
    matrix = estimate_norm(kps, size)
    inverse = _invert_affine(matrix)

    # Pillow samples pixel centers, so shift the affine transform by half a pixel.
    inverse[:, 2] += 0.5 - inverse[:, :2] @ np.full(2, 0.5)
    coefficients = tuple(float(value) for value in inverse.reshape(-1))
    aligned = Image.fromarray(source, mode="RGB").transform(
        (size, size),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BILINEAR,
    )
    return np.array(aligned, dtype=np.uint8, order="C", copy=True)
