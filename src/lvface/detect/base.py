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
        """Detect faces and five-point landmarks in an RGB image.

        Args:
            image: Source RGB uint8 image.

        Returns:
            Detected faces in backend order.
        """

    def align(self, image: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
        """Align five landmarks to the canonical ArcFace template.

        Args:
            image: Source RGB uint8 image.
            kps: Five facial landmarks with shape ``(5, 2)``.
            size: Output crop size.

        Returns:
            The aligned RGB crop.
        """
        return norm_crop(image, kps, size)

    def crop(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect and align every face that has landmarks.

        Args:
            image: Source RGB uint8 image.

        Returns:
            Aligned crops for faces with five-point landmarks.
        """
        self.load()
        return [self.align(image, face.kps) for face in self.detect(image) if face.kps is not None]

    def detect_and_align(self, image: np.ndarray) -> list[Face]:
        """Detect faces and attach crops when landmarks are available.

        Args:
            image: Source RGB uint8 image.

        Returns:
            Detected faces, with aligned crops where possible.
        """
        self.load()
        faces = self.detect(image)

        for face in faces:
            if face.kps is not None:
                face.aligned = self.align(image, face.kps)

        return faces
