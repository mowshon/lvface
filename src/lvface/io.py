"""Image loading and normalization."""

from __future__ import annotations

import ipaddress
import os
import socket
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias
from urllib.parse import urljoin, urlsplit

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

# Pixel cap is enforced per decode below, not via Pillow's process-wide global.
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_REDIRECTS = 5
HTTP_TIMEOUT = (3.05, 10.0)

ImageInput: TypeAlias = str | bytes | os.PathLike[str] | np.ndarray[Any, np.dtype[Any]]


def _validate_array(
    image: np.ndarray[Any, np.dtype[Any]],
) -> npt.NDArray[np.uint8]:
    """Validate and copy an image array as RGB uint8.

    Args:
        image: Grayscale, RGB, or RGBA uint8 image array.

    Returns:
        An owned contiguous RGB uint8 array.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"image array must have dtype uint8, got {image.dtype}")

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image array must have shape (H, W), (H, W, 3), or (H, W, 4), got {image.shape}"
        )

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must have non-zero dimensions, got {image.shape}")

    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(f"decoded image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit")

    return np.array(image, dtype=np.uint8, order="C", copy=True)


def _decode_image(source: BinaryIO) -> npt.NDArray[np.uint8]:
    """Decode a binary image stream as RGB uint8.

    Args:
        source: Readable binary stream containing encoded image data.

    Returns:
        An owned RGB uint8 image.
    """
    try:
        with Image.open(source) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"image must have non-zero dimensions, got {(height, width)}")

            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"decoded image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit")

            return np.array(image.convert("RGB"), dtype=np.uint8, order="C", copy=True)
    except UnidentifiedImageError as error:
        raise ValueError("input is not a valid image") from error
    except (Image.DecompressionBombError, OSError) as error:
        raise ValueError(f"image could not be decoded: {error}") from error


def _validate_url(url: str) -> None:
    """Reject malformed URLs and hosts resolving to non-public addresses.

    Args:
        url: HTTP or HTTPS image URL.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("image URL must use http or https")

    if parsed.hostname is None:
        raise ValueError("image URL must include a hostname")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ValueError(f"image URL hostname could not be resolved: {parsed.hostname}") from error

    if not addresses:
        raise ValueError(f"image URL hostname could not be resolved: {parsed.hostname}")

    # Check every DNS answer; a later connection check also guards rebinding.
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError(f"image URL resolves to a non-public address: {ip}")


def _validate_connected_socket(sock: Any) -> Any:
    """Ensure a connected socket reached a public IP address.

    Args:
        sock: Connected socket-like object.

    Returns:
        The original socket when its peer address is public.
    """
    ip = ipaddress.ip_address(sock.getpeername()[0])
    if ip.is_global:
        return sock
    sock.close()
    raise OSError(f"image URL connected to a non-public address: {ip}")


def _public_address_adapter(requests: Any) -> Any:
    """Build a requests adapter that rejects private peer addresses.

    Args:
        requests: Imported ``requests`` module.

    Returns:
        A configured HTTP adapter instance.
    """
    from urllib3 import PoolManager
    from urllib3.connection import HTTPConnection, HTTPSConnection
    from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

    class PublicHTTPConnection(HTTPConnection):
        def _new_conn(self) -> Any:
            """Open and validate a public HTTP connection."""
            return _validate_connected_socket(super()._new_conn())

    class PublicHTTPSConnection(HTTPSConnection):
        def _new_conn(self) -> Any:
            """Open and validate a public HTTPS connection."""
            return _validate_connected_socket(super()._new_conn())

    class PublicHTTPConnectionPool(HTTPConnectionPool):
        ConnectionCls = PublicHTTPConnection

    class PublicHTTPSConnectionPool(HTTPSConnectionPool):
        ConnectionCls = PublicHTTPSConnection

    class PublicPoolManager(PoolManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize pools using public-address connection classes.

            Args:
                *args: Positional options forwarded to ``PoolManager``.
                **kwargs: Keyword options forwarded to ``PoolManager``.
            """
            super().__init__(*args, **kwargs)
            self.pool_classes_by_scheme = {
                "http": PublicHTTPConnectionPool,
                "https": PublicHTTPSConnectionPool,
            }

    class PublicAddressAdapter(requests.adapters.HTTPAdapter):  # type: ignore[misc]
        def init_poolmanager(
            self,
            connections: int,
            maxsize: int,
            block: bool = False,
            **pool_kwargs: Any,
        ) -> None:
            """Initialize the guarded urllib3 pool manager.

            Args:
                connections: Number of connection pools to cache.
                maxsize: Maximum connections retained per pool.
                block: Whether to block when a pool is exhausted.
                **pool_kwargs: Additional pool-manager options.
            """
            self.poolmanager = PublicPoolManager(
                num_pools=connections,
                maxsize=maxsize,
                block=block,
                **pool_kwargs,
            )

    return PublicAddressAdapter()


def _download_image(url: str) -> bytes:
    """Download an image while enforcing redirect and size limits.

    Args:
        url: Public HTTP or HTTPS image URL.

    Returns:
        Downloaded encoded image bytes.
    """
    try:
        import requests
    except ModuleNotFoundError as error:
        if error.name == "requests":
            raise ImportError('URL image loading requires `pip install "lvface[http]"`') from error
        raise

    session = requests.Session()
    session.trust_env = False
    adapter = _public_address_adapter(requests)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    current_url = url
    redirect_count = 0

    try:
        while True:
            _validate_url(current_url)
            try:
                response = session.get(
                    current_url,
                    allow_redirects=False,
                    stream=True,
                    timeout=HTTP_TIMEOUT,
                )
                response.raise_for_status()
            except requests.RequestException as error:
                raise ValueError(f"image URL request failed: {error}") from error
            with response:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("image URL redirect is missing a Location header")

                    if redirect_count == MAX_REDIRECTS:
                        raise ValueError(f"image URL exceeded {MAX_REDIRECTS} redirects")

                    current_url = urljoin(current_url, location)
                    redirect_count += 1
                    continue

                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError as error:
                        raise ValueError("image URL returned an invalid Content-Length") from error

                    if declared_size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"image URL exceeds the {MAX_DOWNLOAD_BYTES:,}-byte download limit"
                        )

                data = bytearray()
                try:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue

                        data.extend(chunk)
                        if len(data) > MAX_DOWNLOAD_BYTES:
                            raise ValueError(
                                f"image URL exceeds the {MAX_DOWNLOAD_BYTES:,}-byte download limit"
                            )
                except requests.RequestException as error:
                    raise ValueError(f"image URL request failed: {error}") from error
                return bytes(data)
    finally:
        session.close()


def _load_path(value: str) -> npt.NDArray[np.uint8]:
    """Open and decode an image from a filesystem path.

    Args:
        value: Image path.

    Returns:
        Decoded RGB uint8 image.
    """
    try:
        with Path(value).expanduser().open("rb") as file:
            return _decode_image(file)
    except OSError as error:
        raise ValueError(f"image path could not be opened: {value}") from error


def load_image(src: ImageInput) -> npt.NDArray[np.uint8]:
    """Load an image as RGB uint8.

    NumPy inputs are assumed to already use RGB channel order. URL safeguards reduce common
    SSRF and resource-exhaustion risks, but services accepting untrusted URLs should also enforce
    network egress controls.

    Args:
        src: File path, HTTP URL, encoded bytes, or image array.

    Returns:
        An owned RGB uint8 image array.
    """
    if isinstance(src, np.ndarray):
        return _validate_array(src)

    if isinstance(src, bytes):
        return _decode_image(BytesIO(src))

    if not isinstance(src, (str, os.PathLike)):
        raise TypeError("image source must be a path, URL, bytes, or NumPy array")

    if isinstance(src, os.PathLike):
        return _load_path(os.fsdecode(src))

    value = os.fsdecode(src)
    scheme = urlsplit(value).scheme
    if scheme in {"http", "https"}:
        return _decode_image(BytesIO(_download_image(value)))

    if scheme:
        if Path(value).expanduser().is_file():
            return _load_path(value)
        raise ValueError(f"image source must be an existing file or an http or https URL: {value}")

    return _load_path(value)
