import hashlib
import os
from pathlib import Path

import pytest

from lvface import registry


def tiny_model(content: bytes = b"weights") -> registry.Model:
    return registry.Model(
        repo_id="owner/repo",
        filename="tiny.onnx",
        revision="abc123",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        dim=512,
    )


def test_registry_contains_the_four_pinned_released_models() -> None:
    assert registry.DEFAULT_MODEL == "LVFace-T_Glint360K"
    assert set(registry.MODELS) == {
        "LVFace-T_Glint360K",
        "LVFace-S_Glint360K",
        "LVFace-B_Glint360K",
        "LVFace-L_Glint360K",
    }
    assert {model.revision for model in registry.MODELS.values()} == {
        "83b567cd6a3fc34434667e4415b6125feceb39ea"
    }
    assert {model.repo_id for model in registry.MODELS.values()} == {"Mowshon/lvface-weights"}
    assert all(model.dim == 512 for model in registry.MODELS.values())


def test_default_cache_dir_uses_override_or_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LVFACE_CACHE", str(tmp_path))
    assert registry.default_cache_dir() == tmp_path

    monkeypatch.delenv("LVFACE_CACHE")
    monkeypatch.setattr("lvface.registry.Path.home", lambda: tmp_path)
    assert registry.default_cache_dir() == tmp_path / ".cache" / "lvface"


def test_model_cache_path_and_unknown_model(tmp_path: Path) -> None:
    path = registry.model_cache_path(registry.DEFAULT_MODEL, cache_dir=tmp_path)

    assert path == tmp_path / registry.DEFAULT_MODEL / "LVFace-T_Glint360K.onnx"
    with pytest.raises(ValueError, match="unknown"):
        registry.model_cache_path("missing", cache_dir=tmp_path)


def test_validate_model_file_checks_presence_size_and_digest(tmp_path: Path) -> None:
    model = tiny_model()
    valid = tmp_path / model.filename
    valid.write_bytes(b"weights")

    assert registry.validate_model_file(valid, model) == valid.resolve()
    with pytest.raises(FileNotFoundError, match="not found"):
        registry.validate_model_file(tmp_path / "missing.onnx", model)

    valid.write_bytes(b"short")
    with pytest.raises(ValueError, match="size mismatch"):
        registry.validate_model_file(valid, model)

    invalid_digest = tiny_model(b"xxxxxxx")
    valid.write_bytes(b"weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        registry.validate_model_file(valid, invalid_digest)


def test_validate_model_file_reuses_cached_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    path = tmp_path / model.filename
    path.write_bytes(b"weights")

    assert registry.validate_model_file(path, model) == path.resolve()

    def fail_hash(*args: object, **kwargs: object) -> object:
        raise AssertionError("a validated, unchanged file must not be re-hashed")

    monkeypatch.setattr(registry.hashlib, "sha256", fail_hash)
    assert registry.validate_model_file(path, model) == path.resolve()


def test_validate_model_file_rechecks_a_replaced_file_with_preserved_metadata(
    tmp_path: Path,
) -> None:
    model = tiny_model()
    path = tmp_path / model.filename
    path.write_bytes(b"weights")
    original = path.stat()

    assert registry.validate_model_file(path, model) == path.resolve()

    replacement = tmp_path / "replacement.onnx"
    replacement.write_bytes(b"WEIGHTS")
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        registry.validate_model_file(path, model)


def test_validate_model_file_rejects_replacement_during_cached_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    path = tmp_path / model.filename
    path.write_bytes(b"weights")

    assert registry.validate_model_file(path, model) == path.resolve()
    real_validation_key = registry._validation_key
    calls = 0

    def changing_validation_key(
        resolved: Path,
        stat: os.stat_result,
        expected: registry.Model,
    ) -> tuple[str, int, int, int, int, int, str]:
        nonlocal calls
        calls += 1
        key = real_validation_key(resolved, stat, expected)
        if calls == 2:
            return (*key[:-1], "changed")
        return key

    monkeypatch.setattr(registry, "_validation_key", changing_validation_key)

    with pytest.raises(ValueError, match="changed during validation"):
        registry.validate_model_file(path, model)


def test_validate_model_file_rejects_changes_during_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    path = tmp_path / model.filename
    path.write_bytes(b"weights")
    real_fstat = registry.os.fstat
    calls = 0

    def changing_fstat(fd: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        stat = real_fstat(fd)
        if calls == 2:
            values = list(stat)
            values[8] += 1
            return os.stat_result(values)
        return stat

    monkeypatch.setattr(registry.os, "fstat", changing_fstat)

    with pytest.raises(ValueError, match="changed during validation"):
        registry.validate_model_file(path, model)


def test_validate_model_file_rejects_path_changes_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    path = tmp_path / model.filename
    path.write_bytes(b"weights")
    real_open = Path.open
    real_stat = Path.stat
    opened = False

    def tracking_open(self: Path, *args: object, **kwargs: object) -> object:
        nonlocal opened
        file = real_open(self, *args, **kwargs)
        if self == path:
            opened = True
        return file

    def changed_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        stat = real_stat(self, *args, **kwargs)
        if opened and self == path:
            values = list(stat)
            values[8] += 1
            return os.stat_result(values)
        return stat

    monkeypatch.setattr(Path, "open", tracking_open)
    monkeypatch.setattr(Path, "stat", changed_stat)

    with pytest.raises(ValueError, match="changed during validation"):
        registry.validate_model_file(path, model)


def test_validate_model_file_allows_platform_specific_descriptor_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    path = tmp_path / model.filename
    path.write_bytes(b"weights")
    real_fstat = registry.os.fstat

    def distinct_fstat(fd: int) -> os.stat_result:
        stat = real_fstat(fd)
        values = list(stat)
        values[1] += 1
        return os.stat_result(values)

    monkeypatch.setattr(registry.os, "fstat", distinct_fstat)

    assert registry.validate_model_file(path, model) == path.resolve()


def test_validate_model_file_rejects_path_changes_during_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    path = tmp_path / model.filename
    path.write_bytes(b"weights")
    real_sha256 = registry.hashlib.sha256
    real_stat = Path.stat
    hashing_started = False

    class ChangingHash:
        def __init__(self) -> None:
            self.digest = real_sha256()

        def update(self, chunk: bytes) -> None:
            nonlocal hashing_started
            self.digest.update(chunk)
            hashing_started = True

        def hexdigest(self) -> str:
            return self.digest.hexdigest()

    def changed_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        stat = real_stat(self, *args, **kwargs)
        if hashing_started and self == path:
            values = list(stat)
            values[8] += 1
            return os.stat_result(values)
        return stat

    monkeypatch.setattr(registry.hashlib, "sha256", ChangingHash)
    monkeypatch.setattr(Path, "stat", changed_stat)

    with pytest.raises(ValueError, match="changed during validation"):
        registry.validate_model_file(path, model)


def test_resolve_model_path_accepts_explicit_file(tmp_path: Path) -> None:
    model_path = tmp_path / "custom.onnx"
    model_path.write_bytes(b"custom")

    assert registry.resolve_model_path(model_path) == model_path.resolve()
    assert registry.resolve_model_path(str(model_path)) == model_path.resolve()


def test_resolve_model_path_validates_warm_package_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    monkeypatch.setitem(registry.MODELS, "Tiny", model)
    cached = tmp_path / "Tiny" / model.filename
    cached.parent.mkdir()
    cached.write_bytes(b"weights")

    assert registry.resolve_model_path("Tiny", cache_dir=tmp_path) == cached.resolve()


def test_resolve_model_path_reports_missing_or_unknown_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(registry.MODELS, "Tiny", tiny_model())

    with pytest.raises(FileNotFoundError, match=r"lvface\[hub\]"):
        registry.resolve_model_path("Tiny", cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        registry.resolve_model_path(tmp_path / "missing.onnx")
    with pytest.raises(ValueError, match="unknown"):
        registry.resolve_model_path("not-a-model")
