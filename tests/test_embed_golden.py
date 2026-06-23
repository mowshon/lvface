import json
import os
from pathlib import Path

import numpy as np
import pytest

from lvface.embed import LVFaceOnnxEmbedder
from lvface.metrics import cosine
from lvface.registry import DEFAULT_MODEL, MODELS, resolve_model_path

DATA_DIR = Path(__file__).parent / "data"


def test_frozen_preprocessing_parity_and_provenance() -> None:
    crop = np.load(DATA_DIR / "golden_crop_T.npy")
    golden = np.load(DATA_DIR / "golden_pre_T.npy")
    metadata = json.loads((DATA_DIR / "golden_meta.json").read_text(encoding="utf-8"))

    actual = LVFaceOnnxEmbedder("unused.onnx").preprocess(crop)

    np.testing.assert_array_equal(actual, golden)
    assert metadata == {
        "crop_color_order": "RGB",
        "model": DEFAULT_MODEL,
        "mirror_repo_id": MODELS[DEFAULT_MODEL].repo_id,
        "mirror_revision": MODELS[DEFAULT_MODEL].revision,
        "official_repo_id": "bytedance-research/LVFace",
        "official_revision": "b12702ab1f5c721748e054a66dc90e1edd1f0724",
        "model_sha256": MODELS[DEFAULT_MODEL].sha256,
        "preprocessing": "RGB_CHW_x_div_255_minus_0.5_div_0.5_v1",
        "generator": "legacy LVFaceONNXInferencer._preprocess_image",
    }


def _t_model_path() -> Path:
    configured = os.environ.get("LVFACE_TEST_MODEL")
    if configured:
        return Path(configured)
    try:
        return resolve_model_path(DEFAULT_MODEL)
    except FileNotFoundError:
        pytest.skip("T model is not present in the package cache")


@pytest.mark.model
def test_frozen_raw_embedding_and_public_api_parity() -> None:
    crop = np.load(DATA_DIR / "golden_crop_T.npy")
    golden_raw = np.load(DATA_DIR / "golden_raw_T.npy")
    embedder = LVFaceOnnxEmbedder(_t_model_path(), device="cpu")
    preprocessed = embedder.preprocess(crop)
    embedder.load()

    actual_raw = embedder._forward(preprocessed[None, ...])
    raw_embedding = embedder.embed(crop, normalize=False)
    normalized_embedding = embedder.embed(crop)

    np.testing.assert_allclose(actual_raw, golden_raw, atol=1e-4, rtol=0.0)
    assert cosine(actual_raw[0], golden_raw[0]) > 0.9999
    np.testing.assert_allclose(raw_embedding.vector, golden_raw[0], atol=1e-4, rtol=0.0)
    assert not raw_embedding.normalized
    assert normalized_embedding.normalized
    assert np.linalg.norm(normalized_embedding.vector) == pytest.approx(1.0)


@pytest.mark.model
def test_batch_inference_matches_per_image_inference() -> None:
    crop = np.load(DATA_DIR / "golden_crop_T.npy")
    crops = [crop, np.roll(crop, 1, axis=0), np.roll(crop, 1, axis=1)]
    embedder = LVFaceOnnxEmbedder(_t_model_path(), device="cpu")

    per_image = [embedder.embed(item, normalize=False).vector for item in crops]
    batched = embedder.embed_batch(crops, normalize=False, batch_size=len(crops))

    for expected, actual in zip(per_image, batched, strict=True):
        np.testing.assert_allclose(actual.vector, expected, atol=1e-5, rtol=0.0)


@pytest.mark.model
@pytest.mark.parametrize("model_name", MODELS)
def test_cached_released_models_produce_finite_embeddings(model_name: str) -> None:
    configured = os.environ.get("LVFACE_TEST_MODEL")
    if model_name == DEFAULT_MODEL and configured:
        path = Path(configured)
    else:
        try:
            path = resolve_model_path(model_name)
        except FileNotFoundError:
            pytest.skip(f"{model_name} is not present in the package cache")
    crop = np.load(DATA_DIR / "golden_crop_T.npy")

    embedding = LVFaceOnnxEmbedder(path, device="cpu").embed(crop, normalize=False)

    assert embedding.vector.shape == (MODELS[model_name].dim,)
    assert np.isfinite(embedding.vector).all()
