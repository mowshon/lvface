from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from lvface.embed.onnx import LVFaceOnnxEmbedder, _resolve_providers


class Metadata:
    def __init__(self, name: str, shape: list[object], type_: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_


class FakeSession:
    def __init__(
        self,
        *,
        inputs: list[Metadata] | None = None,
        outputs: list[Metadata] | None = None,
        started: threading.Event | None = None,
        finish: threading.Event | None = None,
    ) -> None:
        self.inputs = [Metadata("data", ["batch", 3, 112, 112])] if inputs is None else inputs
        self.outputs = [Metadata("embedding", ["batch", 512])] if outputs is None else outputs
        self.started = started
        self.finish = finish
        self.runs: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[Metadata]:
        return self.inputs

    def get_outputs(self) -> list[Metadata]:
        return self.outputs

    def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.runs.append((names, feeds))
        batch = next(iter(feeds.values()))
        return [np.ones((len(batch), 512), dtype=np.float32)]


@pytest.mark.parametrize(
    ("device", "available", "expected"),
    [
        ("cpu", ["CPUExecutionProvider"], ["CPUExecutionProvider"]),
        (
            "auto",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        ("auto", ["CPUExecutionProvider"], ["CPUExecutionProvider"]),
    ],
)
def test_provider_resolution(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    available: list[str],
    expected: list[str],
) -> None:
    monkeypatch.setattr(
        "lvface.embed.onnx.onnxruntime.get_available_providers",
        lambda: available,
    )

    assert _resolve_providers(device) == expected


def test_cuda_request_warns_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lvface.embed.onnx.onnxruntime.get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    with pytest.warns(RuntimeWarning, match="unavailable"):
        assert _resolve_providers("cuda") == ["CPUExecutionProvider"]


def test_provider_resolution_rejects_invalid_device_or_missing_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="device"):
        _resolve_providers("metal")
    monkeypatch.setattr(
        "lvface.embed.onnx.onnxruntime.get_available_providers",
        lambda: ["CUDAExecutionProvider"],
    )
    with pytest.raises(RuntimeError, match="CPUExecutionProvider"):
        _resolve_providers("auto")


def install_session_factory(
    monkeypatch: pytest.MonkeyPatch,
    session: FakeSession,
    calls: list[tuple[str, list[str]]],
) -> None:
    def factory(path: str, *, providers: list[str]) -> FakeSession:
        calls.append((path, providers))
        return session

    monkeypatch.setattr("lvface.embed.onnx.onnxruntime.InferenceSession", factory)
    monkeypatch.setattr(
        "lvface.embed.onnx.onnxruntime.get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )


def test_load_is_lazy_idempotent_and_forward_uses_discovered_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    session = FakeSession()
    calls: list[tuple[str, list[str]]] = []
    install_session_factory(monkeypatch, session, calls)
    embedder = LVFaceOnnxEmbedder(model_path, device="cpu")

    embedding = embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8), normalize=False)
    embedder.load()

    assert len(calls) == 1
    assert calls[0] == (str(model_path), ["CPUExecutionProvider"])
    assert embedding.vector.shape == (512,)
    names, feeds = session.runs[0]
    assert names == ["embedding"]
    assert list(feeds) == ["data"]


def test_fixed_batch_metadata_drives_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    session = FakeSession(
        inputs=[Metadata("data", [2, 3, 112, 112])],
        outputs=[Metadata("embedding", [2, 512])],
    )
    calls: list[tuple[str, list[str]]] = []
    install_session_factory(monkeypatch, session, calls)
    embedder = LVFaceOnnxEmbedder(model_path)

    result = embedder.embed_batch(
        [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(3)],
        normalize=False,
    )

    assert len(result) == 3
    assert [len(next(iter(feeds.values()))) for _, feeds in session.runs] == [2, 2]


@pytest.mark.parametrize(
    ("inputs", "outputs", "message"),
    [
        ([], [Metadata("out", ["batch", 512])], "one input and one output"),
        (
            [Metadata("a", ["batch", 3, 112, 112]), Metadata("b", ["batch", 3, 112, 112])],
            [Metadata("out", ["batch", 512])],
            "one input and one output",
        ),
        (
            [Metadata("data", ["batch", 3, 112, 112], "tensor(double)")],
            [Metadata("out", ["batch", 512])],
            "float32",
        ),
        (
            [Metadata("data", [3, 112, 112])],
            [Metadata("out", ["batch", 512])],
            "rank 4",
        ),
        (
            [Metadata("data", ["batch", 3, 224, 224])],
            [Metadata("out", ["batch", 512])],
            "must have shape",
        ),
        (
            [Metadata("data", [0, 3, 112, 112])],
            [Metadata("out", [0, 512])],
            "invalid fixed batch",
        ),
        (
            [Metadata("data", [object(), 3, 112, 112])],
            [Metadata("out", ["batch", 512])],
            "unsupported batch axis",
        ),
        (
            [Metadata("data", ["batch", 3, 112, 112])],
            [Metadata("out", ["batch", 256])],
            "model output",
        ),
        (
            [Metadata("data", ["batch", 3, 112, 112])],
            [Metadata("out", ["batch", 512], "tensor(double)")],
            "output must be float32",
        ),
        (
            [Metadata("data", ["batch", 3, 112, 112])],
            [Metadata("out", [512])],
            "model output",
        ),
        (
            [Metadata("data", ["batch", 3, 112, 112])],
            [Metadata("out", [0, 512])],
            "invalid output batch",
        ),
        (
            [Metadata("data", ["batch", 3, 112, 112])],
            [Metadata("out", [1, 512])],
            "fixed batch size",
        ),
        (
            [Metadata("data", [2, 3, 112, 112])],
            [Metadata("out", [3, 512])],
            "batch sizes differ",
        ),
        (
            [Metadata("data", ["batch", 3, 112, 112])],
            [Metadata("out", [object(), 512])],
            "unsupported output batch axis",
        ),
    ],
)
def test_load_rejects_invalid_model_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inputs: list[Metadata],
    outputs: list[Metadata],
    message: str,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    session = FakeSession(inputs=inputs, outputs=outputs)
    install_session_factory(monkeypatch, session, [])

    with pytest.raises(ValueError, match=message):
        LVFaceOnnxEmbedder(model_path).load()


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        LVFaceOnnxEmbedder(tmp_path / "missing.onnx").load()


def test_forward_requires_a_loaded_session(tmp_path: Path) -> None:
    embedder = LVFaceOnnxEmbedder(tmp_path / "model.onnx")

    with pytest.raises(RuntimeError, match="loaded"):
        embedder._forward(np.zeros((1, 3, 112, 112), dtype=np.float32))


def test_concurrent_load_builds_only_one_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    factory_started = threading.Event()
    allow_finish = threading.Event()
    calls: list[int] = []

    def factory(path: str, *, providers: list[str]) -> FakeSession:
        del path, providers
        calls.append(1)
        factory_started.set()
        assert allow_finish.wait(timeout=5)
        return FakeSession()

    monkeypatch.setattr("lvface.embed.onnx.onnxruntime.InferenceSession", factory)
    monkeypatch.setattr(
        "lvface.embed.onnx.onnxruntime.get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    embedder = LVFaceOnnxEmbedder(model_path)
    threads = [threading.Thread(target=embedder.load) for _ in range(2)]
    threads[0].start()
    assert factory_started.wait(timeout=5)
    threads[1].start()
    allow_finish.set()
    for thread in threads:
        thread.join(timeout=5)

    assert calls == [1]
    assert all(not thread.is_alive() for thread in threads)
