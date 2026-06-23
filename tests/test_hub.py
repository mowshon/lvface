from __future__ import annotations

import builtins
import hashlib
import logging
import sys
import types
from pathlib import Path

import pytest

from lvface import hub, registry


def tiny_model(content: bytes = b"weights") -> registry.Model:
    return registry.Model(
        repo_id="Mowshon/lvface-weights",
        filename="tiny.onnx",
        revision="83b567cd6a3fc34434667e4415b6125feceb39ea",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )


@pytest.fixture(autouse=True)
def reset_license_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub, "_license_notice_logged", False)


def test_explicit_file_bypasses_hub_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "custom.onnx"
    model_path.write_bytes(b"custom")
    monkeypatch.setattr(
        hub,
        "_hf_hub_download",
        lambda: pytest.fail("Hugging Face must not be imported"),
    )

    assert hub.resolve_weights(model_path) == model_path.resolve()
    assert hub.resolve_weights(str(model_path)) == model_path.resolve()


def test_missing_path_and_unknown_name_bypass_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hub,
        "_hf_hub_download",
        lambda: pytest.fail("Hugging Face must not be imported"),
    )

    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        hub.resolve_weights(tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        hub.resolve_weights("models/missing")
    with pytest.raises(ValueError, match="registered models"):
        hub.resolve_weights("not-a-model")


def test_valid_package_cache_bypasses_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    monkeypatch.setitem(registry.MODELS, "Tiny", model)
    monkeypatch.setenv("LVFACE_CACHE", str(tmp_path))
    cached = registry.model_cache_path("Tiny")
    cached.parent.mkdir()
    cached.write_bytes(b"weights")
    monkeypatch.setattr(
        hub,
        "_hf_hub_download",
        lambda: pytest.fail("Hugging Face must not be imported"),
    )

    assert hub.resolve_weights("Tiny") == cached.resolve()


def test_invalid_package_cache_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    monkeypatch.setitem(registry.MODELS, "Tiny", model)
    monkeypatch.setenv("LVFACE_CACHE", str(tmp_path))
    cached = registry.model_cache_path("Tiny")
    cached.parent.mkdir()
    cached.write_bytes(b"bad")
    monkeypatch.setattr(
        hub,
        "_hf_hub_download",
        lambda: pytest.fail("invalid cache must not trigger a download"),
    )

    with pytest.raises(ValueError, match="size mismatch"):
        hub.resolve_weights("Tiny")


def test_download_uses_pinned_root_coordinates_and_reuses_hf_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = tiny_model()
    monkeypatch.setitem(registry.MODELS, "Tiny", model)
    monkeypatch.setenv("LVFACE_CACHE", str(tmp_path / "lvface"))
    downloaded = tmp_path / "hf" / model.filename
    downloaded.parent.mkdir()
    downloaded.write_bytes(b"weights")
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(hub, "_hf_hub_download", lambda: fake_download)

    with caplog.at_level(logging.WARNING, logger="lvface.hub"):
        assert hub.resolve_weights("Tiny", cache_dir=tmp_path / "hf-cache") == downloaded.resolve()
        assert hub.resolve_weights("Tiny", cache_dir=tmp_path / "hf-cache") == downloaded.resolve()

    assert calls == [
        {
            "repo_id": "Mowshon/lvface-weights",
            "filename": "tiny.onnx",
            "revision": "83b567cd6a3fc34434667e4415b6125feceb39ea",
            "cache_dir": tmp_path / "hf-cache",
        },
        {
            "repo_id": "Mowshon/lvface-weights",
            "filename": "tiny.onnx",
            "revision": "83b567cd6a3fc34434667e4415b6125feceb39ea",
            "cache_dir": tmp_path / "hf-cache",
        },
    ]
    assert all("subfolder" not in call for call in calls)
    assert caplog.text.count("weight licensing is unresolved") == 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"bad", "size mismatch"),
        (b"WEIGHTS", "checksum mismatch"),
    ],
)
def test_invalid_download_is_rejected(
    content: bytes,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    monkeypatch.setitem(registry.MODELS, "Tiny", model)
    monkeypatch.setenv("LVFACE_CACHE", str(tmp_path / "lvface"))
    downloaded = tmp_path / model.filename
    downloaded.write_bytes(content)
    monkeypatch.setattr(hub, "_hf_hub_download", lambda: lambda **kwargs: str(downloaded))

    with pytest.raises(ValueError, match=message):
        hub.resolve_weights("Tiny")


def test_missing_hub_dependency_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(registry.MODELS, "Tiny", tiny_model())
    monkeypatch.setenv("LVFACE_CACHE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ImportError, match=r"lvface\[hub\].*explicit local"):
        hub.resolve_weights("Tiny")


def test_hub_import_preserves_nested_dependency_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fail_nested_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "huggingface_hub":
            error = ModuleNotFoundError("missing nested dependency")
            error.name = "httpx"
            raise error
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_nested_import)

    with pytest.raises(ModuleNotFoundError, match="nested dependency"):
        hub._hf_hub_download()


def test_hub_import_returns_download_function(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(**kwargs: object) -> str:
        return "cached.onnx"

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.hf_hub_download = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    assert hub._hf_hub_download() is fake_download


def test_license_notice_handles_concurrent_prior_logging(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class LoggingLock:
        def __enter__(self) -> None:
            monkeypatch.setattr(hub, "_license_notice_logged", True)

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(hub, "_license_notice_lock", LoggingLock())

    with caplog.at_level(logging.WARNING, logger="lvface.hub"):
        hub._log_license_notice()

    assert caplog.text == ""
