"""Download and resolve released LVFace ONNX weights."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

from lvface.registry import MODELS, model_cache_path, validate_model_file

_LICENSE_NOTICE = (
    "LVFace weight licensing is unresolved: the official repository metadata says MIT, while "
    "its model-card prose restricts downloaded models to non-commercial research. Weights are "
    "downloaded from the unofficial Mowshon/lvface-weights preservation mirror, which grants "
    "no additional rights. Review the official model card and cite the LVFace paper."
)
_license_notice_lock = threading.Lock()
_license_notice_logged = False
logger = logging.getLogger(__name__)


def _looks_like_path(model: str | os.PathLike[str], candidate: Path) -> bool:
    """Return whether a model argument appears to be a file path.

    Args:
        model: Original model argument.
        candidate: Path representation of the argument.

    Returns:
        ``True`` when the argument should be treated as a path.
    """
    if not isinstance(model, str):
        return True

    return (
        candidate.suffix.lower() == ".onnx"
        or candidate.parent != Path(".")
        or "/" in model
        or "\\" in model
    )


def _hf_hub_download() -> Callable[..., str]:
    """Import the optional Hugging Face download function.

    Returns:
        The ``hf_hub_download`` callable.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as error:
        if error.name == "huggingface_hub":
            raise ImportError(
                'automatic model downloads require `pip install "lvface[hub]"`; '
                "alternatively pass an explicit local .onnx path"
            ) from error
        raise
    return cast(Callable[..., str], hf_hub_download)


def _log_license_notice() -> None:
    """Log the LVFace weight-license notice once per process."""
    global _license_notice_logged
    if _license_notice_logged:
        return

    with _license_notice_lock:
        if _license_notice_logged:
            return

        logger.warning(_LICENSE_NOTICE)
        _license_notice_logged = True


def resolve_weights(
    model: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve an ONNX path, downloading registered weights when needed.

    The LVFace weight license remains unresolved: official metadata says MIT, while official
    prose limits downloaded models to non-commercial research. The unofficial preservation
    mirror grants no additional rights. The mirror is pinned at
    ``83b567cd6a3fc34434667e4415b6125feceb39ea`` and preserves files from the official
    ``bytedance-research/LVFace`` revision
    ``b12702ab1f5c721748e054a66dc90e1edd1f0724``. Review the official model card and cite the
    LVFace paper.

    Args:
        model: Explicit ONNX path or registered model name.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        Resolved path to a validated ONNX model.
    """
    candidate = Path(model).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    name = os.fspath(model)
    if _looks_like_path(model, candidate):
        raise FileNotFoundError(f"ONNX model not found: {candidate}")

    if name not in MODELS:
        registered = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown LVFace model: {name!r}; registered models: {registered}")

    registered_model = MODELS[name]
    cached = model_cache_path(name)
    if cached.is_file():
        return validate_model_file(cached, registered_model)

    # Download only after local resolution and validation have been exhausted.
    download = _hf_hub_download()
    _log_license_notice()
    downloaded = download(
        repo_id=registered_model.repo_id,
        filename=registered_model.filename,
        revision=registered_model.revision,
        cache_dir=cache_dir,
    )
    return validate_model_file(downloaded, registered_model)
