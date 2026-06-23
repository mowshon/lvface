from __future__ import annotations

import builtins
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lvface import io as image_io
from lvface import load_image


def encoded_image(mode: str, size: tuple[int, int] = (3, 2)) -> bytes:
    buffer = BytesIO()
    Image.new(mode, size, 128).save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_accepts_path_bytes_and_rgb_array(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(encoded_image("RGB"))
    array = np.full((2, 3, 3), 17, dtype=np.uint8)

    from_path = load_image(path)
    from_string_path = load_image(str(path))
    from_bytes = load_image(path.read_bytes())
    from_array = load_image(array)

    assert from_path.shape == (2, 3, 3)
    np.testing.assert_array_equal(from_path, from_string_path)
    np.testing.assert_array_equal(from_path, from_bytes)
    np.testing.assert_array_equal(from_array, array)
    assert not np.shares_memory(from_array, array)


def test_load_image_accepts_paths_containing_a_colon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "image:copy.png"
    path.write_bytes(encoded_image("RGB"))
    monkeypatch.chdir(tmp_path)

    assert load_image(path).shape == (2, 3, 3)
    assert load_image(path.name).shape == (2, 3, 3)


def test_load_image_coerces_pillow_and_array_channels_to_rgb() -> None:
    grayscale = load_image(encoded_image("L"))
    rgba = load_image(encoded_image("RGBA"))
    array_gray = load_image(np.full((2, 3), 9, dtype=np.uint8))
    array_rgba = load_image(np.full((2, 3, 4), 11, dtype=np.uint8))

    assert grayscale.shape == rgba.shape == array_gray.shape == array_rgba.shape == (2, 3, 3)
    np.testing.assert_array_equal(array_gray, 9)
    np.testing.assert_array_equal(array_rgba, 11)


@pytest.mark.parametrize(
    ("source", "error", "message"),
    [
        (np.zeros((2, 3, 3), dtype=np.float32), ValueError, "uint8"),
        (np.zeros((2, 3, 2), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((0, 3, 3), dtype=np.uint8), ValueError, "non-zero"),
        (object(), TypeError, "path, URL, bytes"),
        (b"not an image", ValueError, "valid image"),
        ("file:///tmp/image.png", ValueError, "http or https"),
    ],
)
def test_load_image_rejects_invalid_inputs(
    source: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        load_image(source)  # type: ignore[arg-type]


def test_load_image_enforces_pixel_cap_for_arrays_and_decoded_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(image_io, "MAX_IMAGE_PIXELS", 4)

    with pytest.raises(ValueError, match="pixel limit"):
        load_image(np.zeros((2, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="pixel limit"):
        load_image(encoded_image("RGB"))


def test_load_image_reports_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not be opened"):
        load_image(tmp_path / "missing.png")


def test_decode_reports_zero_dimensions_and_decoder_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ZeroImage:
        size = (0, 1)

        def __enter__(self) -> ZeroImage:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(image_io.Image, "open", lambda source: ZeroImage())
    with pytest.raises(ValueError, match="non-zero"):
        image_io._decode_image(BytesIO(b"unused"))

    def decoder_error(source: object) -> object:
        raise OSError("truncated")

    monkeypatch.setattr(image_io.Image, "open", decoder_error)
    with pytest.raises(ValueError, match="could not be decoded"):
        image_io._decode_image(BytesIO(b"unused"))


def test_url_validation_blocks_non_public_and_unresolvable_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_io.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("127.0.0.1", 80))],
    )
    with pytest.raises(ValueError, match="non-public"):
        image_io._validate_url("https://example.com/image.png")

    def fail_resolution(*args: object, **kwargs: object) -> list[object]:
        raise image_io.socket.gaierror("no host")

    monkeypatch.setattr(image_io.socket, "getaddrinfo", fail_resolution)
    with pytest.raises(ValueError, match="could not be resolved"):
        image_io._validate_url("https://example.com/image.png")

    monkeypatch.setattr(image_io.socket, "getaddrinfo", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="could not be resolved"):
        image_io._validate_url("https://example.com/image.png")


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("file:///tmp/a", "http or https"),
        ("https:///image.png", "hostname"),
    ],
)
def test_url_validation_rejects_invalid_urls(url: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        image_io._validate_url(url)


class FakeResponse:
    def __init__(
        self,
        *,
        chunks: list[bytes] | None = None,
        headers: dict[str, str] | None = None,
        redirect: bool = False,
        permanent_redirect: bool = False,
    ) -> None:
        self.chunks = chunks or []
        self.headers = headers or {}
        self.is_redirect = redirect
        self.is_permanent_redirect = permanent_redirect

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int) -> list[bytes]:
        assert chunk_size == 64 * 1024
        return self.chunks

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeSession:
    responses: list[FakeResponse] = []
    calls: list[tuple[str, object]] = []
    mounts: list[tuple[str, object]] = []
    closed = False

    def __init__(self) -> None:
        self.trust_env = True

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def mount(self, prefix: str, adapter: object) -> None:
        self.mounts.append((prefix, adapter))

    def close(self) -> None:
        type(self).closed = True


def install_fake_session(monkeypatch: pytest.MonkeyPatch, responses: list[FakeResponse]) -> None:
    import requests

    FakeSession.responses = responses
    FakeSession.calls = []
    FakeSession.mounts = []
    FakeSession.closed = False
    monkeypatch.setattr(requests, "Session", FakeSession)


def test_url_loader_streams_with_timeouts_and_decodes_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = encoded_image("RGB")
    install_fake_session(
        monkeypatch,
        [FakeResponse(chunks=[b"", payload], headers={"Content-Length": str(len(payload))})],
    )
    monkeypatch.setattr(image_io, "_validate_url", lambda url: None)

    loaded = load_image("https://example.com/image.png")

    assert loaded.shape == (2, 3, 3)
    assert FakeSession.closed
    assert [prefix for prefix, _ in FakeSession.mounts] == ["http://", "https://"]
    assert FakeSession.calls == [
        (
            "https://example.com/image.png",
            {
                "allow_redirects": False,
                "stream": True,
                "timeout": image_io.HTTP_TIMEOUT,
            },
        )
    ]


def test_url_transport_rejects_a_rebound_private_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests
    from urllib3.connection import HTTPConnection, HTTPSConnection

    class FakeSocket:
        def __init__(self, address: str) -> None:
            self.address = address
            self.closed = False

        def getpeername(self) -> tuple[str, int]:
            return self.address, 443

        def close(self) -> None:
            self.closed = True

    public_socket = FakeSocket("93.184.216.34")
    private_socket = FakeSocket("127.0.0.1")
    adapter = image_io._public_address_adapter(requests)
    http_connection = adapter.poolmanager.pool_classes_by_scheme["http"].ConnectionCls
    https_connection = adapter.poolmanager.pool_classes_by_scheme["https"].ConnectionCls

    monkeypatch.setattr(HTTPConnection, "_new_conn", lambda self: public_socket)
    assert http_connection("example.com")._new_conn() is public_socket

    monkeypatch.setattr(HTTPSConnection, "_new_conn", lambda self: private_socket)
    with pytest.raises(OSError, match="non-public"):
        https_connection("example.com")._new_conn()
    assert private_socket.closed


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            FakeResponse(headers={"Content-Length": str(image_io.MAX_DOWNLOAD_BYTES + 1)}),
            "download limit",
        ),
        (FakeResponse(headers={"Content-Length": "many"}), "invalid Content-Length"),
        (
            FakeResponse(chunks=[b"x" * (image_io.MAX_DOWNLOAD_BYTES + 1)]),
            "download limit",
        ),
    ],
)
def test_url_loader_enforces_declared_and_streamed_size_limits(
    monkeypatch: pytest.MonkeyPatch,
    response: FakeResponse,
    message: str,
) -> None:
    install_fake_session(monkeypatch, [response])
    monkeypatch.setattr(image_io, "_validate_url", lambda url: None)

    with pytest.raises(ValueError, match=message):
        image_io._download_image("https://example.com/image.png")

    assert FakeSession.closed


def test_url_loader_revalidates_redirect_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_session(
        monkeypatch,
        [FakeResponse(headers={"Location": "http://169.254.169.254/latest"}, redirect=True)],
    )
    monkeypatch.setattr(
        image_io.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [
            (2, 1, 6, "", ("93.184.216.34" if host == "example.com" else host, 80))
        ],
    )

    with pytest.raises(ValueError, match="non-public"):
        image_io._download_image("https://example.com/image.png")

    assert len(FakeSession.calls) == 1


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (FakeResponse(redirect=True), "missing a Location"),
        (
            FakeResponse(headers={"Location": "/again"}, permanent_redirect=True),
            "exceeded 0 redirects",
        ),
    ],
)
def test_url_loader_rejects_invalid_or_excessive_redirects(
    monkeypatch: pytest.MonkeyPatch,
    response: FakeResponse,
    message: str,
) -> None:
    install_fake_session(monkeypatch, [response])
    monkeypatch.setattr(image_io, "_validate_url", lambda url: None)
    monkeypatch.setattr(image_io, "MAX_REDIRECTS", 0)

    with pytest.raises(ValueError, match=message):
        image_io._download_image("https://example.com/image.png")


def test_url_loader_wraps_request_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    class FailingSession(FakeSession):
        def get(self, url: str, **kwargs: object) -> FakeResponse:
            raise requests.RequestException("network down")

    monkeypatch.setattr(requests, "Session", FailingSession)
    monkeypatch.setattr(image_io, "_validate_url", lambda url: None)

    with pytest.raises(ValueError, match="request failed"):
        image_io._download_image("https://example.com/image.png")


def test_url_loader_wraps_streaming_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    class FailingStreamResponse(FakeResponse):
        def iter_content(self, *, chunk_size: int) -> list[bytes]:
            raise requests.RequestException("read timed out")

    install_fake_session(monkeypatch, [FailingStreamResponse()])
    monkeypatch.setattr(image_io, "_validate_url", lambda url: None)

    with pytest.raises(ValueError, match="request failed"):
        image_io._download_image("https://example.com/image.png")


def test_url_loader_requires_http_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "requests", None)

    with pytest.raises(ImportError, match=r"lvface\[http\]"):
        image_io._download_image("https://example.com/image.png")


def test_url_loader_does_not_mask_nested_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fail_nested_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name == "requests":
            raise ModuleNotFoundError("missing dependency", name="urllib3")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_nested_import)
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        image_io._download_image("https://example.com/image.png")
