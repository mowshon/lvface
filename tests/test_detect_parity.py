from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lvface.detect.align import ARCFACE_DST, estimate_norm, norm_crop
from lvface.detect.insightface import InsightFaceDetector
from lvface.embed import LVFaceOnnxEmbedder
from lvface.metrics import cosine
from lvface.registry import DEFAULT_MODEL, resolve_model_path

insightface = pytest.importorskip("insightface", exc_type=ImportError)
from insightface.utils import face_align  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def transformed_landmarks() -> np.ndarray:
    landmarks = ARCFACE_DST.copy()
    landmarks[:, 0] = landmarks[:, 0] * 0.91 + 4.0
    landmarks[:, 1] = landmarks[:, 1] * 0.94 + 3.0
    return landmarks


@pytest.mark.detect
def test_transform_matches_insightface_reference() -> None:
    landmarks = transformed_landmarks()

    expected = face_align.estimate_norm(landmarks, image_size=112, mode="arcface")

    np.testing.assert_allclose(estimate_norm(landmarks), expected, atol=1e-4, rtol=0.0)


@pytest.mark.detect
def test_pillow_crop_stays_within_reference_pixel_bounds() -> None:
    image = np.load(DATA_DIR / "golden_crop_T.npy")
    landmarks = transformed_landmarks()

    pillow_crop = norm_crop(image, landmarks)
    reference_crop = face_align.norm_crop(image, landmark=landmarks, image_size=112)
    difference = np.abs(pillow_crop.astype(np.float32) - reference_crop.astype(np.float32)) / 255

    assert float(difference.mean()) <= 2.0 / 255
    assert float(difference.max()) <= 12.0 / 255


@pytest.mark.detect
def test_default_alignment_is_the_reference_warp() -> None:
    image = np.load(DATA_DIR / "golden_crop_T.npy")
    landmarks = transformed_landmarks()
    detector = InsightFaceDetector()
    detector.app = object()
    detector._face_align = face_align

    actual = detector.align(image, landmarks)
    expected = face_align.norm_crop(image, landmark=landmarks, image_size=112)

    np.testing.assert_array_equal(actual, expected)


def _model_path() -> Path:
    configured = os.environ.get("LVFACE_TEST_MODEL")
    if configured:
        return Path(configured)
    try:
        return resolve_model_path(DEFAULT_MODEL)
    except FileNotFoundError:
        pytest.skip("T model is not present in the package cache")


@pytest.mark.detect
@pytest.mark.model
def test_pillow_crop_embedding_matches_reference() -> None:
    image = np.load(DATA_DIR / "golden_crop_T.npy")
    landmarks = transformed_landmarks()
    embedder = LVFaceOnnxEmbedder(_model_path(), device="cpu")

    pillow_embedding = embedder.embed(norm_crop(image, landmarks))
    reference_embedding = embedder.embed(
        face_align.norm_crop(image, landmark=landmarks, image_size=112)
    )

    assert cosine(pillow_embedding, reference_embedding) > 0.999
