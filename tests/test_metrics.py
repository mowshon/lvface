import inspect

import numpy as np
import pytest

from lvface import Embedding, metrics


def test_vector_metric_identities_and_embedding_inputs() -> None:
    vector = Embedding(np.array([3.0, 4.0], dtype=np.float32))

    assert metrics.cosine(vector, vector) == pytest.approx(1.0)
    assert metrics.euclidean(vector, vector) == pytest.approx(0.0)
    assert metrics.euclidean_normalized(vector, vector) == pytest.approx(0.0)
    assert metrics.dot(vector, vector) == pytest.approx(25.0)


def test_normalized_euclidean_cosine_relationship() -> None:
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([-1.0, 4.0, 2.0])

    distance = metrics.euclidean_normalized(left, right)

    assert distance**2 == pytest.approx(2.0 - 2.0 * metrics.cosine(left, right))


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        (np.ones((1, 2)), np.ones(2), "1-D"),
        (np.array([]), np.array([]), "non-empty"),
        (np.ones(2), np.ones(3), "shapes must match"),
        (np.array([1.0, np.nan]), np.ones(2), "finite"),
    ],
)
def test_vector_metrics_reject_invalid_inputs(
    left: np.ndarray,
    right: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        metrics.dot(left, right)


@pytest.mark.parametrize(
    "function",
    [metrics.cosine, metrics.euclidean_normalized],
)
def test_normalized_vector_metrics_reject_zero_norm(function: object) -> None:
    with pytest.raises(ValueError, match="zero or non-finite norm"):
        function(np.zeros(2), np.ones(2))  # type: ignore[operator]


def test_pairwise_cosine_matches_naive_loop() -> None:
    left = np.array([[1.0, 2.0], [3.0, -1.0]])
    right = np.array([[2.0, 1.0], [-2.0, 4.0], [3.0, 3.0]])

    expected = np.array(
        [[metrics.cosine(a, b) for b in right] for a in left],
        dtype=np.float64,
    )

    np.testing.assert_allclose(metrics.pairwise_cosine(left, right), expected)


@pytest.mark.parametrize("metric", ["euclidean", "euclidean_normalized"])
def test_pairwise_distance_matches_naive_loop(metric: str) -> None:
    left = np.array([[1.0, 2.0], [3.0, -1.0]])
    right = np.array([[2.0, 1.0], [-2.0, 4.0], [3.0, 3.0]])
    function = getattr(metrics, metric)
    expected = np.array([[function(a, b) for b in right] for a in left])

    np.testing.assert_allclose(metrics.pairwise_distance(left, right, metric=metric), expected)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        (np.ones(2), np.ones((1, 2)), "2-D"),
        (np.empty((0, 2)), np.ones((1, 2)), "non-empty"),
        (np.ones((1, 2)), np.ones((1, 3)), "feature dimensions"),
        (np.array([[1.0, np.nan]]), np.ones((1, 2)), "finite"),
    ],
)
def test_pairwise_metrics_reject_invalid_inputs(
    left: np.ndarray,
    right: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        metrics.pairwise_cosine(left, right)


def test_pairwise_normalized_metrics_reject_zero_rows() -> None:
    with pytest.raises(ValueError, match="zero or non-finite norm"):
        metrics.pairwise_cosine(np.zeros((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError, match="zero or non-finite norm"):
        metrics.pairwise_distance(
            np.zeros((1, 2)),
            np.ones((1, 2)),
            metric="euclidean_normalized",
        )


def test_pairwise_distance_rejects_cosine() -> None:
    with pytest.raises(ValueError, match="unsupported distance metric"):
        metrics.pairwise_distance(np.ones((1, 2)), np.ones((1, 2)), metric="cosine")


def test_metric_metadata_allows_only_cosine_decisions() -> None:
    assert metrics.METRICS["cosine"].higher_is_better
    assert metrics.METRICS["cosine"].decision
    assert not metrics.METRICS["euclidean"].higher_is_better
    assert not metrics.METRICS["euclidean"].decision
    assert not metrics.METRICS["euclidean_normalized"].decision
    assert set(metrics.DEFAULT_THRESHOLDS) == {"cosine"}


def test_sigmoid_percentage_is_centered_monotonic_and_bounded() -> None:
    threshold = 0.42
    scores = [-100.0, threshold - 0.1, threshold, threshold + 0.1, 100.0]
    percentages = [
        metrics.to_percentage(score, threshold=threshold, sharpness=12.0) for score in scores
    ]

    assert percentages == sorted(percentages)
    assert percentages[0] == pytest.approx(0.0, abs=1e-12)
    assert percentages[2] == pytest.approx(50.0)
    assert percentages[-1] == pytest.approx(100.0)


def test_percentage_default_threshold_and_linear_shape() -> None:
    threshold = metrics.DEFAULT_THRESHOLDS["cosine"]

    assert metrics.to_percentage(threshold) == pytest.approx(50.0)
    assert metrics.to_percentage(-1.0, method="linear") == 0.0
    assert metrics.to_percentage(0.0, method="linear") == 50.0
    assert metrics.to_percentage(1.0, method="linear") == 100.0
    assert metrics.to_percentage(2.0, method="linear") == 100.0
    assert "metric" not in inspect.signature(metrics.to_percentage).parameters


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"score": np.nan}, "score must be finite"),
        ({"score": 0.5, "threshold": np.inf}, "threshold must be finite"),
        ({"score": 0.5, "sharpness": 0.0}, "sharpness"),
        ({"score": 0.5, "sharpness": np.inf}, "sharpness"),
        ({"score": 0.5, "method": "calibrated"}, "unsupported percentage method"),
    ],
)
def test_percentage_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        metrics.to_percentage(**kwargs)  # type: ignore[arg-type]
