from __future__ import annotations

import builtins
import os
import sys
import types
from pathlib import Path

import pytest

from lvface import runtime


class FakeOrt:
    def __init__(self, providers: list[str]) -> None:
        self.providers = providers
        self.preload_calls: list[dict[str, str]] = []

    def get_available_providers(self) -> list[str]:
        return self.providers

    def preload_dlls(self, *, directory: str) -> None:
        self.preload_calls.append({"directory": directory})


def test_resolve_directml_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    ort = FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: ort)

    assert runtime.resolve_ort_providers("directml") == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_import_onnxruntime_returns_imported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("onnxruntime")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    assert runtime._import_onnxruntime() is fake


def test_import_onnxruntime_missing_package_has_backend_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fail_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "onnxruntime":
            error = ModuleNotFoundError("missing onnxruntime")
            error.name = "onnxruntime"
            raise error
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(builtins, "__import__", fail_import)

    with pytest.raises(RuntimeError, match=r"lvface\[cpu\].*lvface\[cuda\]"):
        runtime._import_onnxruntime()


def test_import_onnxruntime_preserves_nested_module_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fail_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "onnxruntime":
            error = ModuleNotFoundError("missing nested dependency")
            error.name = "coloredlogs"
            raise error
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(builtins, "__import__", fail_import)

    with pytest.raises(ModuleNotFoundError, match="nested dependency"):
        runtime._import_onnxruntime()


def test_import_onnxruntime_import_error_has_reinstall_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fail_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "onnxruntime":
            raise ImportError("bad shared library")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(builtins, "__import__", fail_import)

    with pytest.raises(RuntimeError, match="could not be imported"):
        runtime._import_onnxruntime()


def test_site_packages_nvidia_lib_dirs_discovers_nested_lib_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})

    assert runtime._site_packages_nvidia_lib_dirs() == []

    cuda_lib = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    cudnn_lib = tmp_path / "nvidia" / "cudnn" / "lib"
    cuda_lib.mkdir(parents=True)
    cudnn_lib.mkdir(parents=True)

    assert runtime._site_packages_nvidia_lib_dirs() == [cuda_lib, cudnn_lib]


def test_preload_uses_nvidia_site_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: ort)
    monkeypatch.setattr(runtime, "_site_packages_nvidia_lib_dirs", lambda: [])

    runtime.preload_onnxruntime_gpu_dependencies()

    assert ort.preload_calls == [{"directory": ""}]


def test_preload_returns_when_onnxruntime_cannot_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: (_ for _ in ()).throw(RuntimeError))

    runtime.preload_onnxruntime_gpu_dependencies()


def test_preload_suppresses_ort_preload_errors_and_updates_linux_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingPreloadOrt(FakeOrt):
        def preload_dlls(self, *, directory: str) -> None:
            raise RuntimeError("preload failed")

    lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: FailingPreloadOrt([]))
    monkeypatch.setattr(runtime, "_site_packages_nvidia_lib_dirs", lambda: [lib_dir])
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "existing")

    runtime.preload_onnxruntime_gpu_dependencies()

    assert os.environ["LD_LIBRARY_PATH"] == f"{lib_dir}:existing"


def test_preload_updates_windows_dll_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    handles: list[str] = []

    def add_dll_directory(path: str) -> str:
        handles.append(path)
        return f"handle:{path}"

    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: object())
    monkeypatch.setattr(runtime, "_site_packages_nvidia_lib_dirs", lambda: [lib_dir])
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.os, "add_dll_directory", add_dll_directory, raising=False)
    monkeypatch.setattr(runtime, "_DLL_DIRECTORY_HANDLES", [])

    runtime.preload_onnxruntime_gpu_dependencies()

    assert handles == [str(lib_dir)]
    assert [f"handle:{lib_dir}"] == runtime._DLL_DIRECTORY_HANDLES


def test_preload_ignores_non_linux_non_windows_library_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: object())
    monkeypatch.setattr(runtime, "_site_packages_nvidia_lib_dirs", lambda: [lib_dir])
    monkeypatch.setattr(runtime.sys, "platform", "darwin")

    runtime.preload_onnxruntime_gpu_dependencies()


def test_get_available_providers_returns_imported_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: FakeOrt(["CPUExecutionProvider"]))

    assert runtime.get_available_ort_providers() == ["CPUExecutionProvider"]


def test_missing_onnxruntime_has_backend_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> object:
        raise RuntimeError("missing")

    monkeypatch.setattr(runtime, "_import_onnxruntime", missing)

    with pytest.raises(RuntimeError, match=r"lvface\[cpu\].*lvface\[cuda\].*lvface\[directml\]"):
        runtime.resolve_ort_providers("cpu")


def test_available_providers_returns_empty_when_onnxruntime_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> object:
        raise RuntimeError("missing")

    monkeypatch.setattr(runtime, "_import_onnxruntime", missing)

    assert runtime.get_available_ort_providers() == []


def test_resolve_cuda_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: ort)

    assert runtime.resolve_ort_providers("cuda") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_resolve_directml_missing_provider_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    ort = FakeOrt(["CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: ort)

    with pytest.raises(RuntimeError, match="DmlExecutionProvider"):
        runtime.resolve_ort_providers("directml")


def test_resolve_auto_prefers_directml_after_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    ort = FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_import_onnxruntime", lambda: ort)

    assert runtime.resolve_ort_providers("auto") == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
