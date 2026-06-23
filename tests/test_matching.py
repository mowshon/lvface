from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import lvface.recognizer as recognizer_module
from lvface import BBox, Embedding, Face, FaceRecognizer, NoFaceError, metrics


def face(
    vector: tuple[float, float],
    *,
    image_index: int = 0,
    face_index: int = 0,
    size: float = 10.0,
    score: float = 0.5,
) -> Face:
    normalized = np.asarray(vector, dtype=np.float64)
    normalized /= np.linalg.norm(normalized)
    return Face(
        BBox(0.0, 0.0, size, size, score),
        None,
        None,
        Embedding(normalized, normalized=True),
        image_index=image_index,
        face_index=face_index,
    )


def recognizer(
    monkeypatch: pytest.MonkeyPatch,
    analyzed: dict[str, list[Face]],
    *,
    threshold: float = 0.35,
) -> FaceRecognizer:
    instance = object.__new__(FaceRecognizer)
    instance.threshold = threshold
    monkeypatch.setattr(instance, "analyze", lambda source: analyzed[str(source)])
    return instance


def cluster_keys(clusters: list[list[Face]]) -> list[list[tuple[int, int]]]:
    return [[(item.image_index, item.face_index) for item in cluster] for cluster in clusters]


def test_collection_assigns_stable_image_and_face_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = face((1.0, 0.0), image_index=99, face_index=4)
    instance = recognizer(monkeypatch, {"a": [original], "b": [original]})

    collected = instance._collect(["a", "b"])

    assert [(item.image_index, item.face_index) for item in collected] == [(0, 4), (1, 4)]
    assert all(item is not original for item in collected)
    assert original.image_index == 99


def test_collection_treats_encoded_bytes_as_one_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = face((1.0, 0.0))
    instance = recognizer(monkeypatch, {str(b"image"): [original]})

    collected = instance._collect(b"image")

    assert len(collected) == 1
    assert collected[0].image_index == 0


def test_find_selects_query_ranks_gallery_and_filters_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    largest = face((1.0, 0.0), face_index=0, size=20.0, score=0.5)
    highest = face((0.0, 1.0), face_index=1, size=10.0, score=0.9)
    instance = recognizer(
        monkeypatch,
        {
            "query": [largest, highest],
            "g1": [face((0.8, 0.6), face_index=0)],
            "g2": [face((1.0, 0.0), face_index=0)],
            "g3": [face((-1.0, 0.0), face_index=0)],
        },
    )

    with pytest.warns(RuntimeWarning, match="largest"):
        matches = instance.find(
            "query",
            ["g1", "g2", "g3"],
            select="largest",
            top_k=2,
            threshold=0.5,
        )

    assert [match.score for match in matches] == pytest.approx([1.0, 0.8])
    assert [(match.candidate.image_index, match.candidate.face_index) for match in matches] == [
        (1, 0),
        (0, 0),
    ]
    assert all(match.is_match for match in matches)
    assert all(match.query.face_index == 0 for match in matches)

    with pytest.warns(RuntimeWarning, match="highest score"):
        highest_match = instance.find("query", "g1", select="highest_score", threshold=-1.0)
    assert highest_match[0].query.face_index == 1

    with pytest.raises(ValueError, match="exactly one"):
        instance.find("query", "g1", select="error")


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_find_rejects_invalid_top_k(
    monkeypatch: pytest.MonkeyPatch,
    top_k: object,
) -> None:
    instance = recognizer(monkeypatch, {})

    with pytest.raises(ValueError, match="positive integer"):
        instance.find("query", "gallery", top_k=top_k)  # type: ignore[arg-type]


def test_find_handles_empty_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = recognizer(monkeypatch, {"query": [face((1.0, 0.0))], "empty": []})

    assert instance.find("query", "empty") == []
    with pytest.raises(NoFaceError, match="no face"):
        instance.find("empty", "query")


def test_match_matrix_all_and_greedy_assignments(monkeypatch: pytest.MonkeyPatch) -> None:
    query_faces = [face((1.0, 0.0), face_index=0), face((0.0, 1.0), face_index=1)]
    candidate_faces = [
        face((1.0, 0.0), face_index=0),
        face((np.sqrt(0.5), np.sqrt(0.5)), face_index=1),
    ]
    instance = recognizer(monkeypatch, {"query": query_faces, "gallery": candidate_faces})

    all_result = instance.match("query", "gallery", assignment="all", threshold=0.5)
    greedy_result = instance.match("query", "gallery", threshold=0.5)
    expected = metrics.pairwise_cosine(
        np.stack([item.embedding.vector for item in query_faces]),  # type: ignore[union-attr]
        np.stack([item.embedding.vector for item in candidate_faces]),  # type: ignore[union-attr]
    )

    np.testing.assert_allclose(all_result.matrix, expected)
    assert len(all_result.pairs) == 3
    assert [(pair.query.face_index, pair.candidate.face_index) for pair in greedy_result.pairs] == [
        (0, 0),
        (1, 1),
    ]
    assert greedy_result.best_for(0) is greedy_result.pairs[0]


def test_pair_ranking_preserves_close_similarity_order() -> None:
    query_faces = [face((1.0, 0.0), face_index=0)]
    candidate_faces = [
        face((1.0, 0.0), face_index=0),
        face((1.0, 0.0), face_index=1),
    ]
    matrix = np.array([[0.90000001, 0.90000004]])

    indices = FaceRecognizer._ranked_indices(
        matrix,
        query_faces,
        candidate_faces,
        threshold=0.5,
    )

    assert indices == [(0, 1), (0, 0)]


def test_match_validates_assignment_and_handles_empty_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = recognizer(monkeypatch, {"empty": [], "one": [face((1.0, 0.0))]})

    with pytest.raises(ValueError, match="unknown assignment"):
        instance.match("one", "one", assignment="random")  # type: ignore[arg-type]

    left_empty = instance.match("empty", "one")
    right_empty = instance.match("one", "empty")
    assert left_empty.matrix.shape == (0, 1)
    assert right_empty.matrix.shape == (1, 0)
    assert left_empty.pairs == right_empty.pairs == []


def test_similarity_matrix_rejects_missing_embeddings() -> None:
    missing = Face(None, None, None, None)

    with pytest.raises(RuntimeError, match="does not have an embedding"):
        FaceRecognizer._similarity_matrix([missing], [face((1.0, 0.0))])


def test_hungarian_finds_optimal_pairs_and_leaves_weak_face_unmatched() -> None:
    pytest.importorskip("scipy.optimize")
    matrix = np.array(
        [
            [0.90, 0.80],
            [0.85, 0.10],
            [0.20, 0.20],
        ]
    )

    pairs = FaceRecognizer._hungarian_indices(matrix, threshold=0.5)

    assert pairs == [(0, 1), (1, 0)]
    assert all(query_index != 2 for query_index, _ in pairs)


def test_match_dispatches_to_hungarian_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = recognizer(
        monkeypatch,
        {"query": [face((1.0, 0.0))], "gallery": [face((1.0, 0.0))]},
    )
    calls: list[tuple[np.ndarray, float]] = []

    def assign(matrix: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        calls.append((matrix, threshold))
        return [(0, 0)]

    monkeypatch.setattr(FaceRecognizer, "_hungarian_indices", staticmethod(assign))

    result = instance.match("query", "gallery", assignment="hungarian", threshold=0.5)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], [[1.0]])
    assert calls[0][1] == pytest.approx(0.5)
    assert len(result.pairs) == 1
    assert result.pairs[0].score == pytest.approx(1.0)


def test_hungarian_cost_matrix_is_finite_and_prefers_valid_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[np.ndarray] = []

    def solve(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        captured.append(costs)
        return np.array([0, 1]), np.array([0, 1])

    module = SimpleNamespace(linear_sum_assignment=solve)
    monkeypatch.setattr(recognizer_module.importlib, "import_module", lambda name: module)

    pairs = FaceRecognizer._hungarian_indices(np.array([[0.8], [0.1]]), threshold=0.5)

    assert pairs == [(0, 0)]
    assert np.isfinite(captured[0]).all()
    assert captured[0][1, 0] > captured[0][1, 2] + captured[0][2, 0]


def test_hungarian_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError("No module named 'scipy'", name="scipy")

    monkeypatch.setattr(recognizer_module.importlib, "import_module", missing)

    with pytest.raises(ImportError, match=r"lvface\[hungarian\]"):
        FaceRecognizer._hungarian_indices(np.ones((1, 1)), threshold=0.5)


def test_hungarian_reraises_nested_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError("No module named 'dependency'", name="dependency")

    monkeypatch.setattr(recognizer_module.importlib, "import_module", missing)

    with pytest.raises(ModuleNotFoundError, match="dependency"):
        FaceRecognizer._hungarian_indices(np.ones((1, 1)), threshold=0.5)


def test_hungarian_empty_matrix_does_not_require_optional_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def load(name: str) -> object:
        calls.append(name)
        raise AssertionError("empty assignment should not load scipy")

    monkeypatch.setattr(recognizer_module.importlib, "import_module", load)

    assert FaceRecognizer._hungarian_indices(np.empty((0, 1)), threshold=0.5) == []
    assert calls == []


def test_group_blocks_similarity_chaining_and_is_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faces = [
        face((1.0, 0.0), image_index=0),
        face((np.cos(np.pi / 6), np.sin(np.pi / 6)), image_index=1),
        face((np.cos(np.pi / 3), np.sin(np.pi / 3)), image_index=2),
    ]
    instance = object.__new__(FaceRecognizer)
    instance.threshold = 0.8
    orders = iter([faces, [faces[2], faces[0], faces[1]]])
    monkeypatch.setattr(instance, "_collect", lambda images: next(orders))

    first = instance.group(["album"])
    second = instance.group(["album"])

    first_keys = cluster_keys(first)
    assert cluster_keys(second) == first_keys
    assert sorted(map(len, first_keys)) == [1, 2]
    assert not any((0, 0) in cluster and (2, 0) in cluster for cluster in first_keys)


def test_group_enforces_one_face_per_image_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_identity = [
        face((1.0, 0.0), image_index=0, face_index=0),
        face((1.0, 0.0), image_index=0, face_index=1),
    ]
    instance = object.__new__(FaceRecognizer)
    instance.threshold = 0.5
    monkeypatch.setattr(instance, "_collect", lambda images: duplicate_identity)

    separate = instance.group(["collage"])
    together = instance.group(["collage"], one_per_image=False)

    assert cluster_keys(separate) == [[(0, 0)], [(0, 1)]]
    assert cluster_keys(together) == [[(0, 0), (0, 1)]]
    with pytest.raises(TypeError, match="boolean"):
        instance.group(["collage"], one_per_image=1)  # type: ignore[arg-type]


def test_group_skips_edges_within_an_already_merged_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same_identity = [
        face((1.0, 0.0), image_index=0),
        face((1.0, 0.0), image_index=1),
        face((1.0, 0.0), image_index=2),
    ]
    instance = object.__new__(FaceRecognizer)
    instance.threshold = 0.5
    monkeypatch.setattr(instance, "_collect", lambda images: same_identity)

    clusters = instance.group(["album"])

    assert cluster_keys(clusters) == [[(0, 0), (1, 0), (2, 0)]]


def test_group_handles_empty_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = recognizer(monkeypatch, {"empty": []})

    assert instance.group("empty") == []


def test_image_collection_rejects_non_sequence() -> None:
    instance = object.__new__(FaceRecognizer)
    instance.threshold = 0.5

    with pytest.raises(TypeError, match="image source"):
        instance.group(42)  # type: ignore[arg-type]
