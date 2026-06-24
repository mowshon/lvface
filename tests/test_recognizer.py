from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import lvface.hub as hub_module
import lvface.recognizer as recognizer_module
from lvface import (
    BBox,
    Face,
    FaceDetector,
    FaceEmbedder,
    FaceRecognizer,
    NoFaceError,
)
from lvface.detect import ARCFACE_DST
from lvface.registry import DEFAULT_MODEL


class StubEmbedder(FaceEmbedder):
    embedding_dim = 2

    def __init__(self) -> None:
        self.load_calls = 0
        self.forward_calls = 0
        self.loaded = False

    def load(self) -> None:
        if not self.loaded:
            self.loaded = True
            self.load_calls += 1

    def _forward(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.forward_calls += 1
        red = batch[:, 0].mean(axis=(1, 2))
        green = batch[:, 1].mean(axis=(1, 2))
        return np.stack((red + 2.0, green + 2.0), axis=1).astype(np.float32)


class StubDetector(FaceDetector):
    def __init__(self, faces: list[Face]) -> None:
        self.faces = faces
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def detect(self, image: np.ndarray) -> list[Face]:
        return self.faces

    def align(self, image: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
        value = int(kps[0, 0])
        crop = np.zeros((size, size, 3), dtype=np.uint8)
        crop[..., 0] = value
        crop[..., 1] = 255 - value
        return crop


def face(
    *,
    x2: float,
    y2: float,
    score: float,
    landmark_value: float,
    index: int,
) -> Face:
    landmarks = ARCFACE_DST.copy()
    landmarks[0, 0] = landmark_value
    return Face(BBox(0, 0, x2, y2, score), landmarks, None, None, face_index=index)


def image(value: int = 0) -> npt.NDArray[np.uint8]:
    return np.full((112, 112, 3), value, dtype=np.uint8)


def test_analyze_aligns_all_faces_and_embeds_in_one_batch() -> None:
    embedder = StubEmbedder()
    detector = StubDetector(
        [
            face(x2=80, y2=80, score=0.7, landmark_value=10, index=0),
            face(x2=40, y2=40, score=0.9, landmark_value=20, index=1),
            Face(BBox(0, 0, 20, 20, 0.8), None, None, None, face_index=2),
        ]
    )
    recognizer = FaceRecognizer(embedder, detector)

    with pytest.warns(RuntimeWarning, match="landmarks"):
        faces = recognizer.analyze(image())

    assert len(faces) == 2
    assert all(item.aligned is not None and item.embedding is not None for item in faces)
    assert detector.load_calls == 1
    assert embedder.forward_calls == 1


def test_embed_returns_all_or_selects_by_area_and_score() -> None:
    detector = StubDetector(
        [
            face(x2=80, y2=80, score=0.7, landmark_value=10, index=0),
            face(x2=40, y2=40, score=0.9, landmark_value=240, index=1),
        ]
    )
    recognizer = FaceRecognizer(StubEmbedder(), detector)

    assert len(recognizer.embed(image())) == 2
    with pytest.warns(RuntimeWarning, match="largest"):
        largest = recognizer.embed(image(), select="largest")
    with pytest.warns(RuntimeWarning, match="highest score"):
        highest = recognizer.embed(image(), select="highest_score")

    assert largest.vector[0] < largest.vector[1]
    assert highest.vector[0] > highest.vector[1]
    with pytest.raises(ValueError, match="exactly one"):
        recognizer.embed(image(), select="error")


def test_empty_and_invalid_selection_contracts() -> None:
    recognizer = FaceRecognizer(StubEmbedder(), StubDetector([]))

    assert recognizer.analyze(image()) == []
    assert recognizer.embed(image()) == []
    with pytest.raises(NoFaceError, match="no face"):
        recognizer.embed(image(), select="largest")
    with pytest.raises(ValueError, match="unknown face selector"):
        recognizer.embed(image(), select="closest")  # type: ignore[arg-type]


def test_detector_none_embeds_aligned_inputs_and_compare_returns_all_metrics() -> None:
    recognizer = FaceRecognizer(StubEmbedder(), detector=None, threshold=0.5)
    aligned = image(200)

    analyzed = recognizer.analyze(aligned)
    direct = recognizer.embed_aligned(aligned)
    result = recognizer.compare(aligned, aligned)

    assert len(analyzed) == 1
    assert analyzed[0].bbox is None
    np.testing.assert_allclose(analyzed[0].embedding.vector, direct.vector)  # type: ignore[union-attr]
    assert result.cosine == pytest.approx(1.0)
    assert result.euclidean == pytest.approx(0.0)
    assert result.percentage > 50.0
    assert result.is_match
    assert recognizer.verify(aligned, aligned)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metric": "euclidean"}, "diagnostic only"),
        ({"threshold": np.nan}, "threshold"),
        ({"threshold": 2.0}, "between -1 and 1"),
    ],
)
def test_recognizer_rejects_invalid_comparison_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FaceRecognizer(StubEmbedder(), detector=None, **kwargs)  # type: ignore[arg-type]


def test_compare_validates_per_call_metric_and_threshold() -> None:
    recognizer = FaceRecognizer(StubEmbedder(), detector=None)

    with pytest.raises(ValueError, match="diagnostic only"):
        recognizer.compare(image(), image(), metric="euclidean")
    with pytest.raises(ValueError, match="threshold"):
        recognizer.compare(image(), image(), threshold=np.inf)


def test_constructor_resolves_named_adapters_and_install_hints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    resolved: list[object] = []

    def resolve_weights(value: object) -> Path:
        resolved.append(value)
        return model_path

    monkeypatch.setattr(recognizer_module, "resolve_weights", resolve_weights)
    named = FaceRecognizer("Tiny", detector=None, device="directml")

    assert named.embedder.model_path == model_path
    assert named.embedder.device == "directml"
    assert resolved == ["Tiny"]
    with pytest.raises(TypeError, match="embedder"):
        FaceRecognizer(object(), detector=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown face detector"):
        FaceRecognizer(StubEmbedder(), detector="other")
    monkeypatch.setattr(recognizer_module.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"lvface\[detect\]"):
        FaceRecognizer(StubEmbedder(), detector="insightface")
    with pytest.raises(ValueError, match="device"):
        FaceRecognizer(StubEmbedder(), detector=None, device="metal")


def test_constructor_resolves_default_model_and_bypasses_hub_for_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    resolved: list[object] = []

    def resolve_weights(value: object) -> Path:
        resolved.append(value)
        return model_path

    monkeypatch.setattr(recognizer_module, "resolve_weights", resolve_weights)

    default = FaceRecognizer(detector=None, device="cpu")

    assert default.embedder.model_path == model_path
    assert resolved == [DEFAULT_MODEL]

    monkeypatch.undo()
    monkeypatch.setattr(
        hub_module,
        "_hf_hub_download",
        lambda: pytest.fail("explicit paths must bypass Hugging Face"),
    )

    explicit = FaceRecognizer(model_path, detector=None, device="cpu")

    assert explicit.embedder.model_path == model_path.resolve()


def test_constructor_builds_available_named_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = StubDetector([])
    monkeypatch.setattr(recognizer_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(recognizer_module, "InsightFaceDetector", lambda **kwargs: detector)

    recognizer = FaceRecognizer(StubEmbedder(), detector="insightface", device="cuda")

    assert recognizer.detector is detector


def test_embed_defensively_rejects_a_selected_face_without_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer = FaceRecognizer(StubEmbedder(), detector=None)
    monkeypatch.setattr(
        recognizer,
        "analyze",
        lambda src: [Face(None, None, image(), None)],
    )

    with pytest.raises(RuntimeError, match="does not have an embedding"):
        recognizer.embed(image(), select="largest")


def test_selection_requires_boxes_for_detector_faces() -> None:
    recognizer = FaceRecognizer(StubEmbedder(), detector=None)
    faces = [Face(None, None, image(), StubEmbedder().embed(image())) for _ in range(2)]

    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="bounding boxes"):
        recognizer._select(faces, "largest")
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="detection scores"):
        recognizer._select(faces, "highest_score")
