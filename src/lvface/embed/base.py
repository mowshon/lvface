"""Shared face-embedding adapter."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from lvface.types import Embedding


class FaceEmbedder(ABC):
    """Base class for face-embedding backends."""

    input_size: tuple[int, int] = (112, 112)
    embedding_dim: int = 512
    _fixed_batch_size: int | None = None

    @abstractmethod
    def load(self) -> None:
        """Lazily initialize the embedding backend."""

    @abstractmethod
    def _forward(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.floating]:
        """Run a normalized NCHW batch and return raw embeddings."""

    def preprocess(self, crop: np.ndarray) -> npt.NDArray[np.float32]:
        """Convert an aligned RGB uint8 crop to a normalized CHW tensor."""
        expected_shape = (*self.input_size, 3)
        if not isinstance(crop, np.ndarray):
            raise TypeError("crop must be a NumPy array")

        if crop.shape != expected_shape:
            raise ValueError(f"crop must have shape {expected_shape}, got {crop.shape}")

        if crop.dtype != np.uint8:
            raise ValueError(f"crop must have dtype uint8, got {crop.dtype}")

        chw = np.transpose(crop, (2, 0, 1))
        normalized = ((chw / 255.0) - 0.5) / 0.5
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def embed(self, crop: np.ndarray, *, normalize: bool = True) -> Embedding:
        """Embed one aligned RGB crop."""
        return self.embed_batch([crop], normalize=normalize, batch_size=1)[0]

    def embed_batch(
        self,
        crops: Sequence[np.ndarray],
        *,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> list[Embedding]:
        """Embed aligned RGB crops, respecting the backend's batch contract."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        crop_list = list(crops)
        if not crop_list:
            return []

        tensors = [self.preprocess(crop) for crop in crop_list]
        self.load()
        required_batch = self._fixed_batch_size
        chunk_size = batch_size if required_batch is None else required_batch
        vectors: list[npt.NDArray[np.floating]] = []

        for start in range(0, len(tensors), chunk_size):
            chunk = tensors[start : start + chunk_size]
            real_size = len(chunk)

            # Some exported models only accept one fixed batch size.
            if required_batch is not None and real_size < required_batch:
                chunk.extend([chunk[-1]] * (required_batch - real_size))

            batch = np.stack(chunk)
            output = np.asarray(self._forward(batch))
            if output.ndim != 2 or output.shape != (len(chunk), self.embedding_dim):
                raise ValueError(
                    "backend returned invalid embedding shape "
                    f"{output.shape}; expected {(len(chunk), self.embedding_dim)}"
                )

            if not np.issubdtype(output.dtype, np.floating):
                raise ValueError(f"backend returned non-floating embeddings: {output.dtype}")

            if not np.isfinite(output).all():
                raise ValueError("backend returned an embedding containing NaN/Inf")

            vectors.extend(output[:real_size])

        embeddings = [Embedding(vector) for vector in vectors]
        if normalize:
            return [embedding.normalize() for embedding in embeddings]
        return embeddings
