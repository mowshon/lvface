from __future__ import annotations

import builtins
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lvface.detect import insightface as backend
from lvface.detect.align import ARCFACE_DST
from lvface.detect.insightface import InsightFaceDetector


class FakeFaceAlign:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, int]] = []

    def norm_crop(
        self,
        image: np.ndarray,
        *,
        landmark: np.ndarray,
        image_size: int,
    ) -> np.ndarray:
        self.calls.append((image, landmark, image_size))
        return image[:image_size, :image_size].copy()


class FakeAnalysis:
    instances: list[FakeAnalysis] = []
    started: threading.Event | None = None
    finish: threading.Event | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.prepare_calls: list[tuple[int, tuple[int, int]]] = []
        self.images: list[np.ndarray] = []
        self.faces: list[Any] = []
        self.__class__.instances.append(self)
        if self.started is not None:
            self.started.set()
        if self.finish is not None:
            assert self.finish.wait(timeout=5)

    def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
        self.prepare_calls.append((ctx_id, det_size))

    def get(self, image: np.ndarray) -> list[Any]:
        self.images.append(image.copy())
        return self.faces


@pytest.fixture(autouse=True)
def reset_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAnalysis.instances = []
    FakeAnalysis.started = None
    FakeAnalysis.finish = None
    monkeypatch.setattr(backend, "_license_warning_emitted", False)


def install_fake_insightface(monkeypatch: pytest.MonkeyPatch) -> FakeFaceAlign:
    face_align = FakeFaceAlign()
    package = ModuleType("insightface")
    app = ModuleType("insightface.app")
    utils = ModuleType("insightface.utils")
    app.FaceAnalysis = FakeAnalysis  # type: ignore[attr-defined]
    utils.face_align = face_align  # type: ignore[attr-defined]
    package.app = app  # type: ignore[attr-defined]
    package.utils = utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "insightface", package)
    monkeypatch.setitem(sys.modules, "insightface.app", app)
    monkeypatch.setitem(sys.modules, "insightface.utils", utils)
    return face_align


def test_load_is_lazy_idempotent_and_configures_detection_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_insightface(monkeypatch)
    monkeypatch.setattr(
        backend,
        "_resolve_providers",
        lambda device: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    detector = InsightFaceDetector("buffalo_sc", device="cuda", det_size=(320, 240))

    with pytest.warns(UserWarning, match="non-commercial"):
        detector.load()
    detector.load()

    assert len(FakeAnalysis.instances) == 1
    app = FakeAnalysis.instances[0]
    assert app.kwargs == {
        "name": "buffalo_sc",
        "allowed_modules": ["detection"],
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    assert app.prepare_calls == [(0, (320, 240))]


def test_license_warning_is_emitted_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_insightface(monkeypatch)
    monkeypatch.setattr(backend, "_resolve_providers", lambda device: ["CPUExecutionProvider"])

    with pytest.warns(UserWarning, match="non-commercial") as captured:
        InsightFaceDetector().load()
        InsightFaceDetector("buffalo_sc").load()

    assert len(captured) == 1
    assert [item.prepare_calls for item in FakeAnalysis.instances] == [
        [(-1, (640, 640))],
        [(-1, (640, 640))],
    ]


def test_license_warning_handles_a_concurrent_prior_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmittingLock:
        def __enter__(self) -> None:
            monkeypatch.setattr(backend, "_license_warning_emitted", True)

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(backend, "_license_warning_lock", EmittingLock())

    backend._warn_about_model_license()


def test_concurrent_load_builds_one_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_insightface(monkeypatch)
    monkeypatch.setattr(backend, "_resolve_providers", lambda device: ["CPUExecutionProvider"])
    FakeAnalysis.started = threading.Event()
    FakeAnalysis.finish = threading.Event()
    detector = InsightFaceDetector()

    with pytest.warns(UserWarning):
        threads = [threading.Thread(target=detector.load) for _ in range(2)]
        threads[0].start()
        assert FakeAnalysis.started.wait(timeout=5)
        threads[1].start()
        FakeAnalysis.finish.set()
        for thread in threads:
            thread.join(timeout=5)

    assert len(FakeAnalysis.instances) == 1
    assert all(not thread.is_alive() for thread in threads)


def test_detect_swaps_rgb_to_bgr_filters_scores_and_builds_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_insightface(monkeypatch)
    monkeypatch.setattr(backend, "_resolve_providers", lambda device: ["CPUExecutionProvider"])
    detector = InsightFaceDetector(min_score=0.5)
    with pytest.warns(UserWarning):
        detector.load()
    app = FakeAnalysis.instances[0]
    app.faces = [
        SimpleNamespace(
            bbox=np.array([1, 2, 30, 40], dtype=np.float32),
            det_score=np.float32(0.9),
            kps=ARCFACE_DST,
        ),
        SimpleNamespace(
            bbox=np.array([2, 3, 20, 30], dtype=np.float32),
            det_score=0.49,
            kps=ARCFACE_DST,
        ),
        SimpleNamespace(
            bbox=np.array([4, 5, 50, 60], dtype=np.float32),
            det_score=0.8,
        ),
        SimpleNamespace(
            bbox=np.array([4, 5, 50, 60], dtype=np.float32),
            det_score=np.nan,
            kps=ARCFACE_DST,
        ),
    ]
    image = np.zeros((80, 90, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    faces = detector.detect(image)

    assert len(faces) == 2
    assert faces[0].bbox is not None
    assert faces[0].bbox.score == pytest.approx(0.9)
    assert faces[0].kps is not None
    assert faces[0].face_index == 0
    assert faces[1].kps is None
    assert faces[1].face_index == 1
    assert app.images[0].flags.c_contiguous
    assert app.images[0][0, 0].tolist() == [30, 20, 10]
    assert image[0, 0].tolist() == [10, 20, 30]


@pytest.mark.parametrize(
    "bbox",
    [
        np.array([1, 2, 3]),
        np.array([1, 2, 3, np.inf]),
    ],
)
def test_detect_rejects_invalid_backend_bbox(
    monkeypatch: pytest.MonkeyPatch,
    bbox: np.ndarray,
) -> None:
    install_fake_insightface(monkeypatch)
    monkeypatch.setattr(backend, "_resolve_providers", lambda device: ["CPUExecutionProvider"])
    detector = InsightFaceDetector()
    with pytest.warns(UserWarning):
        detector.load()
    FakeAnalysis.instances[0].faces = [SimpleNamespace(bbox=bbox, det_score=0.9, kps=ARCFACE_DST)]

    with pytest.raises(ValueError, match="bounding box"):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "landmarks",
    [
        np.zeros((4, 2), dtype=np.float32),
        np.full((5, 2), np.nan, dtype=np.float32),
    ],
)
def test_detect_rejects_invalid_backend_landmarks(
    monkeypatch: pytest.MonkeyPatch,
    landmarks: np.ndarray,
) -> None:
    install_fake_insightface(monkeypatch)
    monkeypatch.setattr(backend, "_resolve_providers", lambda device: ["CPUExecutionProvider"])
    detector = InsightFaceDetector()
    with pytest.warns(UserWarning):
        detector.load()
    FakeAnalysis.instances[0].faces = [
        SimpleNamespace(
            bbox=np.array([1, 2, 30, 40], dtype=np.float32),
            det_score=0.9,
            kps=landmarks,
        )
    ]

    with pytest.raises(ValueError, match="five-point landmarks"):
        detector.detect(np.zeros((40, 40, 3), dtype=np.uint8))


def test_reference_align_delegates_without_swapping_rgb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    face_align = install_fake_insightface(monkeypatch)
    monkeypatch.setattr(backend, "_resolve_providers", lambda device: ["CPUExecutionProvider"])
    detector = InsightFaceDetector()
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    with pytest.warns(UserWarning):
        crop = detector.align(image, ARCFACE_DST)

    assert crop.dtype == np.uint8
    assert crop[0, 0].tolist() == [10, 20, 30]
    np.testing.assert_array_equal(face_align.calls[0][0], image)
    np.testing.assert_array_equal(face_align.calls[0][1], ARCFACE_DST)
    assert face_align.calls[0][2] == 112
    with pytest.raises(ValueError, match="size=112"):
        detector.align(image, ARCFACE_DST, size=128)


def test_reference_align_rejects_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    face_align = install_fake_insightface(monkeypatch)
    monkeypatch.setattr(backend, "_resolve_providers", lambda device: ["CPUExecutionProvider"])
    detector = InsightFaceDetector()
    detector.app = object()
    detector._face_align = face_align
    face_align.norm_crop = lambda image, *, landmark, image_size: image[:10, :10]  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="returned shape"):
        detector.align(np.zeros((112, 112, 3), dtype=np.uint8), ARCFACE_DST)


def test_missing_insightface_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def import_without_insightface(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "insightface" or name.startswith("insightface."):
            error = ModuleNotFoundError("No module named 'insightface'")
            error.name = "insightface"
            raise error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_insightface)

    with pytest.raises(ImportError, match=r"lvface\[detect\]"):
        backend._import_insightface()


def test_partial_insightface_install_error_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = ModuleType("insightface")
    package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "insightface", package)
    monkeypatch.delitem(sys.modules, "insightface.app", raising=False)

    with pytest.raises(ModuleNotFoundError):
        backend._import_insightface()


def test_transitive_import_error_is_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    error = ModuleNotFoundError("missing transitive package")
    error.name = "missing_dependency"

    def fail_import() -> tuple[type[Any], Any]:
        raise error

    monkeypatch.setattr(backend, "_import_insightface", fail_import)
    monkeypatch.setattr(backend, "_resolve_providers", lambda device: ["CPUExecutionProvider"])

    with pytest.raises(ModuleNotFoundError, match="transitive"):
        InsightFaceDetector().load()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"det_size": (0, 640)},
        {"det_size": (640, -1)},
        {"det_size": (640.0, 640)},
        {"det_size": (True, 640)},
        {"det_size": (640,)},
        {"min_score": -0.1},
        {"min_score": 1.1},
        {"min_score": np.nan},
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        InsightFaceDetector(**kwargs)


def test_defensive_uninitialized_state_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = InsightFaceDetector()
    monkeypatch.setattr(detector, "load", lambda: None)
    image = np.zeros((112, 112, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="initialize"):
        detector.detect(image)
    with pytest.raises(RuntimeError, match="initialize"):
        detector.align(image, ARCFACE_DST)
