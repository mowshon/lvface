"""Use OpenCV YuNet as a custom detector for LVFace."""

from pathlib import Path
from typing import Any

import numpy as np

from lvface import BBox, Face, FaceDetector, FaceRecognizer


class YuNetDetector(FaceDetector):
    """Adapt OpenCV YuNet to the lvface detector interface."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.detector: Any | None = None

    def load(self) -> None:
        if self.detector is not None:
            return

        import cv2

        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (320, 320),
            0.9,
            0.3,
            5000,
        )

    def detect(self, image: np.ndarray) -> list[Face]:
        self.load()
        if self.detector is None:
            return []

        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))

        bgr_image = np.ascontiguousarray(image[:, :, ::-1])
        _, detections = self.detector.detect(bgr_image)

        if detections is None:
            return []

        faces = []
        for row in detections:
            x, y, width, height = row[:4]
            landmarks = row[4:14].reshape(5, 2)

            # YuNet and ArcFace use a different landmark order.
            landmarks = landmarks[[1, 0, 2, 4, 3]]

            faces.append(
                Face(
                    bbox=BBox(x, y, x + width, y + height, row[14]),
                    kps=landmarks,
                    aligned=None,
                    embedding=None,
                    face_index=len(faces),
                )
            )

        return faces


detector = YuNetDetector("face_detection_yunet.onnx")

recognizer = FaceRecognizer(
    embedder="LVFace-T_Glint360K",
    detector=detector,
)

faces = recognizer.analyze("group-photo.jpg")

for face in faces:
    if face.embedding is not None:
        print(face.face_index, face.embedding.vector.shape)
