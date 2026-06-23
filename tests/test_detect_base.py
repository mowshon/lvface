from __future__ import annotations

import numpy as np

from lvface import BBox, Face, FaceDetector
from lvface.detect.align import ARCFACE_DST


class DummyDetector(FaceDetector):
    def __init__(self) -> None:
        self.loaded = False
        self.load_calls = 0
        self.detect_calls = 0

    def load(self) -> None:
        if not self.loaded:
            self.loaded = True
            self.load_calls += 1

    def detect(self, image: np.ndarray) -> list[Face]:
        assert image.shape == (112, 112, 3)
        self.detect_calls += 1
        return [
            Face(BBox(1, 2, 80, 90, 0.9), ARCFACE_DST, None, None, face_index=0),
            Face(BBox(4, 5, 30, 40, 0.8), None, None, None, face_index=1),
        ]


def test_dummy_detector_uses_shared_crop_and_alignment() -> None:
    detector = DummyDetector()
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30

    crops = detector.crop(image)
    faces = detector.detect_and_align(image)

    assert detector.load_calls == 1
    assert detector.detect_calls == 2
    assert len(crops) == 1
    assert crops[0].shape == (112, 112, 3)
    assert crops[0][50, 50].tolist() == [10, 20, 30]
    assert faces[0].aligned is not None
    assert faces[1].aligned is None
