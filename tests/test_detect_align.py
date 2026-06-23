from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import lvface.detect.align as align_module
from lvface.detect.align import (
    ARCFACE_DST,
    _invert_affine,
    _umeyama,
    estimate_norm,
    norm_crop,
)
from lvface.errors import AlignmentError


def test_arcface_template_is_immutable_and_maps_to_itself() -> None:
    with pytest.raises(ValueError):
        ARCFACE_DST[0, 0] = 0.0

    matrix = estimate_norm(ARCFACE_DST)

    np.testing.assert_allclose(matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-5)


def test_estimate_norm_recovers_similarity_transform() -> None:
    angle = np.deg2rad(13.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    forward = np.column_stack((1.3 * rotation, np.array([7.0, -4.0])))
    homogeneous = np.column_stack((ARCFACE_DST, np.ones(5)))
    source = homogeneous @ np.linalg.inv(np.vstack((forward, [0.0, 0.0, 1.0]))).T

    actual = estimate_norm(source[:, :2])

    np.testing.assert_allclose(actual, forward, atol=1e-4)


@pytest.mark.parametrize(
    ("landmarks", "message"),
    [
        (object(), "numeric"),
        (np.zeros((4, 2)), "exactly 5"),
        (np.full((5, 2), np.nan), "finite"),
        (np.column_stack((np.arange(5), np.arange(5))), "degenerate"),
    ],
)
def test_estimate_norm_rejects_invalid_landmarks(landmarks: object, message: str) -> None:
    with pytest.raises(AlignmentError, match=message):
        estimate_norm(landmarks)


def test_estimate_norm_rejects_unsupported_size() -> None:
    with pytest.raises(AlignmentError, match="size=112"):
        estimate_norm(ARCFACE_DST, size=128)


def test_estimate_norm_rejects_failed_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(align_module, "_umeyama", lambda source, destination: None)

    with pytest.raises(AlignmentError, match="degenerate"):
        estimate_norm(ARCFACE_DST)


def test_umeyama_handles_reflection_and_degenerate_inputs() -> None:
    source = ARCFACE_DST.astype(np.float64)
    reflected = source.copy()
    reflected[:, 0] *= -1

    reflected_matrix = _umeyama(source, reflected)
    zero_variance = _umeyama(np.ones((5, 2)), reflected)
    rank_zero = _umeyama(
        np.array([[0, 0], [1, 0], [-1, 0]], dtype=np.float64),
        np.zeros((3, 2), dtype=np.float64),
    )

    assert reflected_matrix is not None
    assert np.isfinite(reflected_matrix).all()
    assert zero_variance is None
    assert rank_zero is None


def test_umeyama_rank_one_branches_are_finite() -> None:
    source = np.array([[-1, 0], [0, 0], [1, 0]], dtype=np.float64)
    same = source.copy()
    reversed_axis = source.copy()
    reversed_axis[:, 0] *= -1

    assert np.isfinite(_umeyama(source, same)).all()
    assert np.isfinite(_umeyama(source, reversed_axis)).all()


def test_invert_affine_rejects_singular_or_nonfinite_transforms() -> None:
    with pytest.raises(AlignmentError, match="invertible"):
        _invert_affine(np.zeros((2, 3)))
    with pytest.raises(AlignmentError, match="not finite"):
        _invert_affine(np.array([[1, 0, np.inf], [0, 1, 0]], dtype=np.float64))


def test_norm_crop_preserves_rgb_and_returns_owned_uint8_array() -> None:
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[..., 0] = np.arange(112, dtype=np.uint8)
    image[..., 1] = np.arange(112, dtype=np.uint8)[:, None]
    image[..., 2] = 211

    crop = norm_crop(image, ARCFACE_DST)

    assert crop.shape == (112, 112, 3)
    assert crop.dtype == np.uint8
    assert crop.flags.c_contiguous
    assert crop.flags.owndata
    assert crop[50, 50].tolist() == pytest.approx(image[50, 50].tolist(), abs=1)


@pytest.mark.detect
def test_norm_crop_stays_within_opencv_pixel_drift_bounds() -> None:
    cv2 = pytest.importorskip("cv2", exc_type=ImportError)
    image = np.load(Path(__file__).parent / "data" / "golden_crop_T.npy")
    landmarks = ARCFACE_DST.copy()
    landmarks[:, 0] = landmarks[:, 0] * 0.91 + 4.0
    landmarks[:, 1] = landmarks[:, 1] * 0.94 + 3.0
    matrix = estimate_norm(landmarks)

    pillow_crop = norm_crop(image, landmarks)
    reference_crop = cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)
    difference = np.abs(pillow_crop.astype(np.float32) - reference_crop.astype(np.float32)) / 255

    assert float(difference.mean()) <= 2.0 / 255
    assert float(difference.max()) <= 12.0 / 255


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        ([[[0, 0, 0]]], TypeError, "NumPy"),
        (np.zeros((0, 112, 3), dtype=np.uint8), ValueError, "non-zero"),
        (np.zeros((112, 0, 3), dtype=np.uint8), ValueError, "non-zero"),
        (np.zeros((112, 112), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((112, 112, 4), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((112, 112, 3), dtype=np.float32), ValueError, "uint8"),
    ],
)
def test_norm_crop_rejects_invalid_images(
    image: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        norm_crop(image, ARCFACE_DST)  # type: ignore[arg-type]
