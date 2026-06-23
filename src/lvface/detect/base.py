"""Shared face detector and aligner adapter."""

from abc import ABC, abstractmethod

import numpy as np

from lvface.types import Face

from .align import norm_crop


class FaceDetector(ABC):
    """Base class for pluggable face detection and alignment backends."""

    @abstractmethod
    def load(self) -> None:
        """Lazily initialize the detection backend."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Face]:
        """Detect faces and five-point landmarks in an RGB image."""

    def align(self, image: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
        """Align five landmarks to the canonical ArcFace template."""
        return norm_crop(image, kps, size)

    def crop(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect and align every face that has five-point landmarks."""
        self.load()
        return [self.align(image, face.kps) for face in self.detect(image) if face.kps is not None]

    def detect_and_align(self, image: np.ndarray) -> list[Face]:
        """Detect faces and attach aligned crops when landmarks are available."""
        self.load()
        faces = self.detect(image)

        for face in faces:
            if face.kps is not None:
                face.aligned = self.align(image, face.kps)

        return faces
