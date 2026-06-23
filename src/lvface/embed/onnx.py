"""ONNX Runtime backend for released LVFace models."""

from __future__ import annotations

import threading
import warnings
from numbers import Integral
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import onnxruntime

from .base import FaceEmbedder

Device = Literal["auto", "cpu", "cuda"]


def _resolve_providers(device: str) -> list[str]:
    """Select available ONNX Runtime providers for a device request.

    Args:
        device: Requested device: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        Providers ordered by preference.
    """
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    available = set(onnxruntime.get_available_providers())
    if "CPUExecutionProvider" not in available:
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")

    if device == "cpu":
        return ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if device == "cuda":
        warnings.warn(
            "CUDAExecutionProvider was requested but is unavailable; using CPUExecutionProvider",
            RuntimeWarning,
            stacklevel=2,
        )
    return ["CPUExecutionProvider"]


class LVFaceOnnxEmbedder(FaceEmbedder):
    """LVFace embedding backend powered by ONNX Runtime."""

    def __init__(self, model_path: str | Path, *, device: Device = "auto") -> None:
        """Configure an ONNX embedding backend.

        Args:
            model_path: Path to the LVFace ONNX model.
            device: Preferred inference device.
        """
        self.model_path = Path(model_path).expanduser()
        self.device = device
        self.session: onnxruntime.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self._load_lock = threading.Lock()
        self._fixed_batch_size = None

    def load(self) -> None:
        """Build and validate the ONNX Runtime session once."""
        if self.session is not None:
            return
        with self._load_lock:
            if self.session is not None:
                return

            if not self.model_path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

            providers = _resolve_providers(self.device)
            session = onnxruntime.InferenceSession(str(self.model_path), providers=providers)
            input_name, output_name, fixed_batch_size = self._validate_io(session)
            self.input_name = input_name
            self.output_name = output_name
            self._fixed_batch_size = fixed_batch_size
            self.session = session

    def _validate_io(
        self,
        session: onnxruntime.InferenceSession,
    ) -> tuple[str, str, int | None]:
        """Validate model input and output metadata.

        Args:
            session: Initialized ONNX Runtime session.

        Returns:
            Input name, output name, and optional fixed batch size.
        """
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"LVFace ONNX model must have one input and one output, got "
                f"{len(inputs)} inputs and {len(outputs)} outputs"
            )

        # Validate the input separately so model errors point to the right side.
        input_meta = inputs[0]
        if input_meta.type != "tensor(float)":
            raise ValueError(f"model input must be float32, got {input_meta.type}")

        input_shape = input_meta.shape
        if len(input_shape) != 4:
            raise ValueError(f"model input must have rank 4, got shape {input_shape}")

        expected_tail = (3, *self.input_size)
        if tuple(input_shape[1:]) != expected_tail:
            raise ValueError(
                f"model input must have shape (N, {expected_tail[0]}, "
                f"{expected_tail[1]}, {expected_tail[2]}), got {input_shape}"
            )

        batch_axis = input_shape[0]
        if isinstance(batch_axis, Integral) and not isinstance(batch_axis, bool):
            fixed_batch_size = int(batch_axis)
            if fixed_batch_size <= 0:
                raise ValueError(f"model has invalid fixed batch size: {fixed_batch_size}")
        elif batch_axis is None or isinstance(batch_axis, str):
            fixed_batch_size = None
        else:
            raise ValueError(f"model has unsupported batch axis: {batch_axis!r}")

        # The output batch contract must agree with the input contract.
        output_meta = outputs[0]
        if output_meta.type != "tensor(float)":
            raise ValueError(f"model output must be float32, got {output_meta.type}")

        output_shape = output_meta.shape
        if len(output_shape) != 2 or output_shape[-1] != self.embedding_dim:
            raise ValueError(
                f"model output must have shape (N, {self.embedding_dim}), got {output_shape}"
            )

        output_batch_axis = output_shape[0]
        if isinstance(output_batch_axis, Integral) and not isinstance(output_batch_axis, bool):
            output_batch_size = int(output_batch_axis)
            if output_batch_size <= 0:
                raise ValueError(f"model has invalid output batch size: {output_batch_size}")

            if fixed_batch_size is None:
                raise ValueError(
                    "model output has a fixed batch size but the input batch axis is dynamic"
                )

            if output_batch_size != fixed_batch_size:
                raise ValueError(
                    "model input and output batch sizes differ: "
                    f"{fixed_batch_size} != {output_batch_size}"
                )
        elif output_batch_axis is not None and not isinstance(output_batch_axis, str):
            raise ValueError(f"model has unsupported output batch axis: {output_batch_axis!r}")

        return input_meta.name, output_meta.name, fixed_batch_size

    def _forward(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.floating]:
        """Run the loaded ONNX model for one preprocessed batch.

        Args:
            batch: Float32 NCHW input batch.

        Returns:
            Raw model embeddings.
        """
        session = self.session
        input_name = self.input_name
        output_name = self.output_name
        if session is None or input_name is None or output_name is None:
            raise RuntimeError("embedder must be loaded before inference")

        output = session.run([output_name], {input_name: batch})[0]
        return cast(npt.NDArray[np.floating], output)
