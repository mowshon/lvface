import numpy as np
import pytest

from lvface import BBox, ComparisonResult, Embedding, Face, Match, MatchResult, metrics


def test_bbox_dimensions() -> None:
    assert BBox(1.0, 2.0, 6.5, 9.0, 0.95).wh == (5.5, 7.0)


@pytest.mark.parametrize(
    "vector",
    [
        np.empty((0,), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        np.array([1.0, np.nan], dtype=np.float32),
        np.array([1.0, np.inf], dtype=np.float32),
    ],
)
def test_embedding_rejects_invalid_vectors(vector: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Embedding(vector)


def test_embedding_requires_truthful_normalized_flag() -> None:
    with pytest.raises(ValueError, match="normalized=True"):
        Embedding(np.array([3.0, 4.0], dtype=np.float32), normalized=True)


def test_embedding_owns_an_immutable_float32_copy() -> None:
    source = np.array([3.0, 4.0], dtype=np.float64)
    embedding = Embedding(source)
    source[0] = 99.0

    assert embedding.vector.dtype == np.float32
    assert embedding.vector.flags.c_contiguous
    assert not embedding.vector.flags.writeable
    np.testing.assert_array_equal(embedding.vector, [3.0, 4.0])
    with pytest.raises(ValueError):
        embedding.vector[0] = 1.0


def test_embedding_normalization_is_valid_and_idempotent() -> None:
    normalized = Embedding(np.array([3.0, 4.0], dtype=np.float32)).normalize()

    assert normalized.normalized
    assert normalized.normalize() is normalized
    assert np.linalg.norm(normalized.vector) == pytest.approx(1.0)


def test_embedding_normalize_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="zero or non-finite norm"):
        Embedding(np.zeros(2, dtype=np.float32)).normalize()


def test_embedding_normalizes_large_finite_values_stably() -> None:
    maximum = np.finfo(np.float32).max
    normalized = Embedding(np.array([maximum, maximum], dtype=np.float32)).normalize()

    np.testing.assert_allclose(normalized.vector, np.sqrt(0.5), rtol=1e-6)


def test_embedding_supports_numpy_array_protocol() -> None:
    embedding = Embedding(np.array([1.0, 2.0], dtype=np.float32))

    converted = np.asarray(embedding, dtype=np.float64)
    uncopied = np.asarray(embedding, copy=False)

    assert converted.dtype == np.float64
    np.testing.assert_array_equal(converted, embedding.vector)
    assert np.shares_memory(uncopied, embedding.vector)


def test_array_backed_types_compare_and_hash_by_identity() -> None:
    first = Embedding(np.array([1.0, 0.0], dtype=np.float32))
    second = Embedding(np.array([1.0, 0.0], dtype=np.float32))
    face = Face(None, None, None, first)
    match = Match(face, face, 0.9, 90.0, True)
    result = MatchResult(np.empty((0, 0)), [], [], [])

    for value in (first, face, match, result):
        assert value == value
        assert {value}
    assert first != second


def test_face_copies_caller_owned_arrays() -> None:
    kps = np.arange(10, dtype=np.float32).reshape(5, 2)
    aligned = np.zeros((112, 112, 3), dtype=np.uint8)
    face = Face(None, kps, aligned, None)

    kps[0, 0] = 99.0
    aligned[0, 0, 0] = 255

    assert face.kps is not None
    assert face.aligned is not None
    assert face.kps[0, 0] == 0.0
    assert face.aligned[0, 0, 0] == 0


def test_face_accepts_absent_arrays_and_result_holds_values() -> None:
    face = Face(None, None, None, None, image_index=2, face_index=3)
    result = ComparisonResult(0.8, 0.4, 95.0, True, 0.35)

    assert (face.image_index, face.face_index) == (2, 3)
    assert result.is_match


def test_match_result_owns_matrix_and_exposes_match_helpers() -> None:
    queries = [
        Face(None, None, None, Embedding(np.array([1.0, 0.0]), normalized=True)),
        Face(None, None, None, Embedding(np.array([0.0, 1.0]), normalized=True)),
    ]
    candidates = [
        Face(None, None, None, Embedding(np.array([1.0, 0.0]), normalized=True)),
        Face(None, None, None, Embedding(np.array([0.0, 1.0]), normalized=True)),
    ]
    best = Match(queries[0], candidates[0], 0.9, 90.0, True)
    weaker = Match(queries[0], candidates[1], 0.8, 80.0, True)
    matrix = np.array([[0.9, 0.8], [0.2, 0.1]])
    result = MatchResult(matrix, queries, candidates, [weaker, best], threshold=0.4)
    matrix[0, 0] = -1.0

    assert result.matrix[0, 0] == pytest.approx(0.9)
    assert not result.matrix.flags.writeable
    assert result.best_for(0) is best
    assert result.best_for(1) is None
    np.testing.assert_allclose(
        result.as_percentage_matrix(),
        [
            [metrics.to_percentage(0.9, threshold=0.4), metrics.to_percentage(0.8, threshold=0.4)],
            [metrics.to_percentage(0.2, threshold=0.4), metrics.to_percentage(0.1, threshold=0.4)],
        ],
    )
    with pytest.raises(IndexError, match="out of range"):
        result.best_for(2)


def test_match_result_uses_default_cosine_threshold() -> None:
    result = MatchResult(np.empty((0, 0)), [], [], [])

    assert result.threshold == metrics.DEFAULT_THRESHOLDS["cosine"]
    assert result.as_percentage_matrix().shape == (0, 0)


@pytest.mark.parametrize(
    ("matrix", "threshold", "message"),
    [
        (np.ones((2, 1)), 0.4, "shape"),
        (np.array([[np.nan]]), 0.4, "NaN/Inf"),
        (np.ones((1, 1)), 2.0, "threshold"),
    ],
)
def test_match_result_rejects_invalid_values(
    matrix: np.ndarray,
    threshold: float,
    message: str,
) -> None:
    face = Face(None, None, None, None)

    with pytest.raises(ValueError, match=message):
        MatchResult(matrix, [face], [face], [], threshold)
