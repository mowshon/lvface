"""InsightFace detection and reference alignment backend."""

from __future__ import annotations

import threading
import warnings
from typing import Any

import numpy as np

from lvface.embed.onnx import Device, _resolve_providers
from lvface.types import BBox, Face

from .align import _validate_image, estimate_norm
from .base import FaceDetector

_LICENSE_WARNING = (
    "InsightFace bundled model packs are licensed for non-commercial research use only. "
    "Use a detector with appropriately licensed weights or pre-aligned crops for other uses."
)
_license_warning_lock = threading.Lock()
_license_warning_emitted = False


def _warn_about_model_license() -> None:
    """Emit the InsightFace model-license warning once per process."""
    global _license_warning_emitted
    if _license_warning_emitted:
        return
    with _license_warning_lock:
        if _license_warning_emitted:
            return
        warnings.warn(_LICENSE_WARNING, UserWarning, stacklevel=3)
        _license_warning_emitted = True


def _import_insightface() -> tuple[type[Any], Any]:
    """Import optional InsightFace detection components.

    Returns:
        The ``FaceAnalysis`` class and face-alignment module.
    """
    try:
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
    except ModuleNotFoundError as error:
        if error.name == "insightface":
            raise ImportError(
                'InsightFace detection requires `pip install "lvface[detect]"`'
            ) from error
        raise
    return FaceAnalysis, face_align


class InsightFaceDetector(FaceDetector):
    """Face detector using InsightFace detection landmarks."""

    def __init__(
        self,
        name: str = "buffalo_l",
        *,
        device: Device = "auto",
        det_size: tuple[int, int] = (640, 640),
        min_score: float = 0.5,
    ) -> None:
        """Configure the InsightFace detector.

        Args:
            name: InsightFace model-pack name.
            device: Preferred inference device.
            det_size: Detection input width and height.
            min_score: Minimum confidence retained as a face.
        """
        if not name:
            raise ValueError("name must not be empty")

        if (
            len(det_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in det_size)
            or any(value <= 0 for value in det_size)
        ):
            raise ValueError("det_size must contain two positive integers")

        if not np.isfinite(min_score) or not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be finite and between 0 and 1")

        self.name = name
        self.device = device
        self.det_size = det_size
        self.min_score = float(min_score)
        self.app: Any | None = None
        self._face_align: Any | None = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """Initialize the requested InsightFace detection pack once."""
        if self.app is not None:
            return
        with self._load_lock:
            if self.app is not None:
                return

            providers = _resolve_providers(self.device)
            face_analysis, face_align = _import_insightface()
            _warn_about_model_license()
            app = face_analysis(
                name=self.name,
                allowed_modules=["detection"],
                providers=providers,
            )
            ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
            app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            self._face_align = face_align
            self.app = app

    def detect(self, image: np.ndarray) -> list[Face]:
        """Detect faces with InsightFace.

        Args:
            image: Source RGB uint8 image.

        Returns:
            Faces meeting the configured confidence threshold.
        """
        source = _validate_image(image)
        self.load()
        if self.app is None:
            raise RuntimeError("detector failed to initialize")

        bgr = np.ascontiguousarray(source[:, :, ::-1])
        detected = self.app.get(bgr)
        faces: list[Face] = []

        for detected_face in detected:
            score = float(detected_face.det_score)
            if not np.isfinite(score) or score < self.min_score:
                continue

            coordinates = np.asarray(detected_face.bbox, dtype=np.float64)
            if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
                raise ValueError("InsightFace returned an invalid bounding box")

            landmarks = getattr(detected_face, "kps", None)
            if landmarks is not None:
                landmarks = np.asarray(landmarks, dtype=np.float32)
                if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
                    raise ValueError("InsightFace returned invalid five-point landmarks")

            faces.append(
                Face(
                    bbox=BBox(
                        float(coordinates[0]),
                        float(coordinates[1]),
                        float(coordinates[2]),
                        float(coordinates[3]),
                        score,
                    ),
                    kps=landmarks,
                    aligned=None,
                    embedding=None,
                    face_index=len(faces),
                )
            )
        return faces

    def align(self, image: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
        """Align an RGB image with InsightFace's reference warp.

        Args:
            image: Source RGB uint8 image.
            kps: Five facial landmarks with shape ``(5, 2)``.
            size: Output crop size.

        Returns:
            An aligned RGB uint8 crop.
        """
        source = _validate_image(image)
        estimate_norm(kps, size)
        self.load()
        if self._face_align is None:
            raise RuntimeError("detector failed to initialize")

        aligned = np.asarray(self._face_align.norm_crop(source, landmark=kps, image_size=size))
        result = _validate_image(aligned)
        expected_shape = (size, size, 3)

        if result.shape != expected_shape:
            raise ValueError(
                f"InsightFace alignment returned shape {result.shape}, expected {expected_shape}"
            )
        return result
