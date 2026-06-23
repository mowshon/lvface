"""Registry and local resolution for released LVFace ONNX weights."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    """Coordinates and integrity metadata for a released model."""

    repo_id: str
    filename: str
    revision: str
    sha256: str
    size: int
    dim: int = 512


_REVISION = "83b567cd6a3fc34434667e4415b6125feceb39ea"
_REPOSITORY = "Mowshon/lvface-weights"

MODELS = {
    "LVFace-T_Glint360K": Model(
        _REPOSITORY,
        "LVFace-T_Glint360K.onnx",
        _REVISION,
        "bf8da0e1e93c432d9a1d874a9ba0990f5859f970e8864b3990f2f33d11f9cdb3",
        76_653_813,
    ),
    "LVFace-S_Glint360K": Model(
        _REPOSITORY,
        "LVFace-S_Glint360K.onnx",
        _REVISION,
        "cd09f27c82ce0a3633fb8b1966d779a7171b23aa4f14ca0de6edf9677573d119",
        304_196_926,
    ),
    "LVFace-B_Glint360K": Model(
        _REPOSITORY,
        "LVFace-B_Glint360K.onnx",
        _REVISION,
        "9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382",
        455_533_594,
    ),
    "LVFace-L_Glint360K": Model(
        _REPOSITORY,
        "LVFace-L_Glint360K.onnx",
        _REVISION,
        "49389036a4a5b69e0efcddfe34839ac72c7a71ce6b4dc1b6821e2ac368c87063",
        1_022_938_188,
    ),
}

DEFAULT_MODEL = "LVFace-T_Glint360K"


def default_cache_dir() -> Path:
    """Return the configured or default model cache directory."""
    configured = os.environ.get("LVFACE_CACHE")
    if configured:
        return Path(configured).expanduser()

    return Path.home() / ".cache" / "lvface"


def model_cache_path(name: str, *, cache_dir: str | Path | None = None) -> Path:
    """Return the cache path for a registered model.

    Args:
        name: Registered model name.
        cache_dir: Optional cache root override.

    Returns:
        Expected path to the model file.
    """
    try:
        model = MODELS[name]
    except KeyError as error:
        raise ValueError(f"unknown LVFace model: {name!r}") from error

    root = default_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
    return root / name / model.filename


_validated_files: dict[tuple[str, int, int, int, int, int, str], Path] = {}


def _validation_key(
    path: Path,
    stat: os.stat_result,
    model: Model,
) -> tuple[str, int, int, int, int, int, str]:
    """Build a cache key from file identity and model metadata.

    Args:
        path: Resolved model path.
        stat: Current file metadata.
        model: Expected model metadata.

    Returns:
        A key that changes when the file or expected digest changes.
    """
    return (
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        model.sha256,
    )


def validate_model_file(path: str | Path, model: Model) -> Path:
    """Validate a model file's size and SHA-256 digest.

    Args:
        path: Model file to validate.
        model: Expected model metadata.

    Returns:
        The resolved validated path.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"model file not found: {resolved}")

    # Keep path and descriptor metadata in separate comparison domains. On Windows,
    # stat() and fstat() can describe the same file with different identity fields.
    path_stat_before = resolved.stat()
    cache_key = _validation_key(resolved, path_stat_before, model)
    cached = _validated_files.get(cache_key)

    if cached is not None:
        if _validation_key(resolved, resolved.stat(), model) != cache_key:
            raise ValueError(f"model file changed during validation: {resolved}")
        return cached

    with resolved.open("rb") as file:
        stat_before = os.fstat(file.fileno())
        descriptor_key = _validation_key(resolved, stat_before, model)
        if _validation_key(resolved, resolved.stat(), model) != cache_key:
            raise ValueError(f"model file changed during validation: {resolved}")

        if stat_before.st_size != model.size:
            raise ValueError(
                f"model size mismatch for {resolved}: "
                f"expected {model.size}, got {stat_before.st_size}"
            )

        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

        stat_after = os.fstat(file.fileno())

    # Re-stat the path as well as the open file to catch replacements mid-read.
    if (
        _validation_key(resolved, stat_after, model) != descriptor_key
        or _validation_key(resolved, resolved.stat(), model) != cache_key
    ):
        raise ValueError(f"model file changed during validation: {resolved}")

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != model.sha256:
        raise ValueError(
            f"model checksum mismatch for {resolved}: expected {model.sha256}, got {actual_sha256}"
        )

    _validated_files[cache_key] = resolved
    return resolved


def resolve_model_path(
    model: str | os.PathLike[str],
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve an explicit ONNX path or a locally cached model.

    Args:
        model: Explicit file path or registered model name.
        cache_dir: Optional cache root override.

    Returns:
        Resolved path to a valid model file.
    """
    candidate = Path(model).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    name = os.fspath(model)
    if name not in MODELS:
        if candidate.suffix == ".onnx" or candidate.parent != Path("."):
            raise FileNotFoundError(f"ONNX model not found: {candidate}")

        raise ValueError(f"unknown LVFace model: {name!r}")

    cached = model_cache_path(name, cache_dir=cache_dir)
    if cached.is_file():
        return validate_model_file(cached, MODELS[name])

    raise FileNotFoundError(
        f"weights for {name!r} were not found at {cached}; pass an explicit local ONNX path "
        'or install the download support with `pip install "lvface[hub]"`'
    )
