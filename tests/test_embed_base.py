from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from lvface.embed import FaceEmbedder


class StubEmbedder(FaceEmbedder):
    def __init__(
        self,
        *,
        fixed_batch_size: int | None = None,
        output: Literal["valid", "shape", "integer", "nan"] = "valid",
    ) -> None:
        self._fixed_batch_size = fixed_batch_size
        self.output = output
        self.loaded = False
        self.load_calls = 0
        self.batches: list[npt.NDArray[np.float32]] = []

    def load(self) -> None:
        if not self.loaded:
            self.loaded = True
            self.load_calls += 1

    def _forward(self, batch: npt.NDArray[np.float32]) -> np.ndarray:
        self.batches.append(batch.copy())
        if self.output == "shape":
            return np.zeros((len(batch), self.embedding_dim - 1), dtype=np.float32)
        if self.output == "integer":
            return np.zeros((len(batch), self.embedding_dim), dtype=np.int64)
        result = np.repeat(batch[:, :1, :1, :1].reshape(-1, 1), self.embedding_dim, axis=1)
        if self.output == "nan":
            result[0, 0] = np.nan
        return result


def crop(value: int = 0) -> npt.NDArray[np.uint8]:
    return np.full((112, 112, 3), value, dtype=np.uint8)


def test_preprocess_preserves_rgb_order_and_matches_legacy_arithmetic() -> None:
    image = crop()
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30

    tensor = StubEmbedder().preprocess(image)

    expected = ((np.transpose(image, (2, 0, 1)) / 255.0) - 0.5) / 0.5
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_array_equal(tensor, expected.astype(np.float32))
    assert tensor[:, 0, 0].tolist() == pytest.approx(
        [10 / 127.5 - 1.0, 20 / 127.5 - 1.0, 30 / 127.5 - 1.0]
    )


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        ([[[0, 0, 0]]], TypeError, "NumPy"),
        (np.zeros((111, 112, 3), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((112, 112, 3), dtype=np.float32), ValueError, "dtype"),
    ],
)
def test_preprocess_rejects_invalid_crops(
    image: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        StubEmbedder().preprocess(image)  # type: ignore[arg-type]


def test_embed_returns_raw_or_normalized_embedding_and_loads_once() -> None:
    embedder = StubEmbedder()

    raw = embedder.embed(crop(255), normalize=False)
    normalized = embedder.embed(crop(255))

    assert not raw.normalized
    assert raw.vector.shape == (512,)
    np.testing.assert_array_equal(raw.vector, 1.0)
    assert normalized.normalized
    assert np.linalg.norm(normalized.vector) == pytest.approx(1.0)
    assert embedder.load_calls == 1


def test_dynamic_batching_uses_requested_chunk_size() -> None:
    embedder = StubEmbedder()

    embeddings = embedder.embed_batch([crop(value) for value in range(5)], batch_size=2)

    assert len(embeddings) == 5
    assert [len(batch) for batch in embedder.batches] == [2, 2, 1]


@pytest.mark.parametrize(
    ("fixed_batch_size", "expected_batches"),
    [(1, [1, 1, 1, 1]), (3, [3, 3])],
)
def test_fixed_batching_pads_and_trims(
    fixed_batch_size: int,
    expected_batches: list[int],
) -> None:
    embedder = StubEmbedder(fixed_batch_size=fixed_batch_size)

    embeddings = embedder.embed_batch(
        [crop(value) for value in (10, 20, 30, 40)],
        normalize=False,
        batch_size=99,
    )

    assert len(embeddings) == 4
    assert [len(batch) for batch in embedder.batches] == expected_batches
    if fixed_batch_size == 3:
        np.testing.assert_array_equal(embedder.batches[-1][1], embedder.batches[-1][0])
        np.testing.assert_array_equal(embedder.batches[-1][2], embedder.batches[-1][0])


def test_empty_batch_does_not_load_and_batch_size_must_be_positive() -> None:
    embedder = StubEmbedder()

    assert embedder.embed_batch([]) == []
    assert not embedder.loaded
    with pytest.raises(ValueError, match="greater than zero"):
        embedder.embed_batch([crop()], batch_size=0)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("shape", "invalid embedding shape"),
        ("integer", "non-floating"),
        ("nan", "NaN/Inf"),
    ],
)
def test_backend_output_is_validated(
    output: Literal["shape", "integer", "nan"],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StubEmbedder(output=output).embed(crop(), normalize=False)
