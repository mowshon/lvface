"""ONNX Runtime backend selection and GPU dependency loading."""

from __future__ import annotations

import os
import sys
import sysconfig
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

Device = Literal["auto", "cpu", "cuda", "directml"]
ProviderName = Literal[
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
]
_DLL_DIRECTORY_HANDLES: list[Any] = []


def _import_onnxruntime() -> Any:
    """Import ONNX Runtime with an actionable install hint."""
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        if error.name == "onnxruntime":
            raise RuntimeError(
                "ONNX Runtime is not installed. Install exactly one backend extra: "
                "lvface[cpu], lvface[cuda], or lvface[directml]."
            ) from error
        raise
    except ImportError as error:
        raise RuntimeError(
            "ONNX Runtime could not be imported. Reinstall exactly one backend extra: "
            "lvface[cpu], lvface[cuda], or lvface[directml]."
        ) from error

    return ort


def _site_packages_nvidia_lib_dirs() -> list[Path]:
    """Return NVIDIA package library directories from the current environment."""
    purelib = Path(sysconfig.get_paths()["purelib"])
    nvidia_dir = purelib / "nvidia"
    if not nvidia_dir.exists():
        return []

    return [path for path in nvidia_dir.rglob("lib") if path.is_dir()]


def preload_onnxruntime_gpu_dependencies() -> None:
    """Best-effort preload for ONNX Runtime CUDA/cuDNN runtime libraries."""
    try:
        ort = _import_onnxruntime()
    except Exception:
        return

    if hasattr(ort, "preload_dlls"):
        with suppress(Exception):
            ort.preload_dlls(directory="")

    lib_dirs = _site_packages_nvidia_lib_dirs()
    if not lib_dirs:
        return

    if sys.platform.startswith("linux"):
        current = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = ":".join(str(path) for path in lib_dirs)
        os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{current}" if current else new_paths
    elif sys.platform == "win32":
        for path in lib_dirs:
            with suppress(Exception):
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))


def get_available_ort_providers() -> list[str]:
    """Return available ONNX Runtime providers, or an empty list if ORT is missing."""
    try:
        ort = _import_onnxruntime()
        return list(ort.get_available_providers())
    except Exception:
        return []


def resolve_ort_providers(device: str = "auto") -> list[str]:
    """Resolve ONNX Runtime providers for a requested inference device."""
    normalized = device.lower().strip()
    if normalized not in {"auto", "cpu", "cuda", "directml"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'directml'")

    if normalized in {"auto", "cuda"}:
        preload_onnxruntime_gpu_dependencies()

    try:
        available = list(_import_onnxruntime().get_available_providers())
    except RuntimeError as error:
        raise RuntimeError(
            "ONNX Runtime is not available. Install exactly one backend extra: "
            "lvface[cpu], lvface[cuda], or lvface[directml]."
        ) from error

    if "CPUExecutionProvider" not in available:
        raise RuntimeError(
            f"CPUExecutionProvider is not available. Available providers: {available}"
        )

    if normalized == "cpu":
        return ["CPUExecutionProvider"]

    if normalized == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. Install lvface[cuda], verify the "
                "NVIDIA driver/CUDA compatibility, and check CUDA/cuDNN library loading. "
                f"Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if normalized == "directml":
        if "DmlExecutionProvider" not in available:
            raise RuntimeError(
                "DmlExecutionProvider is not available. Install lvface[directml] on Windows "
                "with a DirectX 12-capable GPU. "
                f"Available providers: {available}"
            )
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
