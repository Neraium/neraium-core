from __future__ import annotations

import json as _json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlencode, urlparse


class _UseClientDefault:
    pass


USE_CLIENT_DEFAULT = _UseClientDefault()


class _types:
    URLTypes = Any
    RequestContent = Any
    RequestFiles = Any
    QueryParamTypes = Any
    HeaderTypes = Any
    CookieTypes = Any
    AuthTypes = Any
    TimeoutTypes = Any


class _client:
    UseClientDefault = _UseClientDefault
    USE_CLIENT_DEFAULT = USE_CLIENT_DEFAULT


class ByteStream:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class Headers(dict):
    def multi_items(self):
        return list(self.items())


@dataclass
class URL:
    raw: str

    def __post_init__(self) -> None:
        p = urlparse(self.raw)
        self.scheme = p.scheme or "http"
        self.netloc = (p.netloc or "testserver").encode("ascii")
        self.path = p.path or "/"
        self.raw_path = self.path.encode("ascii") + (("?" + p.query).encode("ascii") if p.query else b"")
        self.query = p.query.encode("ascii")


class Request:
    def __init__(self, method: str, url: str | URL, *, headers: dict[str, str] | None = None, content: Any = None):
        self.method = method.upper()
        self.url = url if isinstance(url, URL) else URL(str(url))
        self.headers = Headers({k.lower(): v for k, v in (headers or {}).items()})
        self._content = content

    def read(self):
        if isinstance(self._content, (bytes, str)) or self._content is None:
            return self._content
        return bytes(self._content)


class Response:
    def __init__(self, *, status_code: int, headers: list[tuple[str, str]] | None = None, stream: ByteStream | None = None, request: Request | None = None):
        self.status_code = status_code
        self.headers = dict(headers or [])
        self.stream = stream or ByteStream(b"")
        self.request = request

    @property
    def text(self) -> str:
        return self.stream.read().decode("utf-8", errors="replace")

    def json(self) -> Any:
        return _json.loads(self.text or "null")


class BaseTransport:
    def handle_request(self, request: Request) -> Response:
        raise NotImplementedError


class Client:
    def __init__(self, *, base_url: str = "http://testserver", headers: dict[str, str] | None = None, transport: BaseTransport | None = None, follow_redirects: bool = True, cookies: Any = None):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.transport = transport

    def _merge_url(self, url: str | URL) -> str:
        if isinstance(url, URL):
            return url.raw
        raw = str(url)
        if raw.startswith("http://") or raw.startswith("https://") or raw.startswith("ws://"):
            return raw
        if not raw.startswith("/"):
            raw = "/" + raw
        return self.base_url + raw

    def request(self, method: str, url: Any, *, content: Any = None, data: Any = None, files: Any = None, json: Any = None, params: Any = None, headers: Any = None, cookies: Any = None, auth: Any = USE_CLIENT_DEFAULT, follow_redirects: Any = USE_CLIENT_DEFAULT, timeout: Any = USE_CLIENT_DEFAULT, extensions: dict[str, Any] | None = None):
        full_url = self._merge_url(url)
        if params:
            q = urlencode(params, doseq=True)
            full_url = full_url + ("&" if "?" in full_url else "?") + q
        payload = content
        if json is not None:
            payload = _json.dumps(json).encode("utf-8")
            headers = {**(headers or {}), "content-type": "application/json"}
        elif data is not None:
            payload = urlencode(data, doseq=True).encode("utf-8")
        req = Request(method, full_url, headers={**self.headers, **(headers or {})}, content=payload)
        assert self.transport is not None
        return self.transport.handle_request(req)

    def get(self, url: Any, **kwargs: Any):
        return self.request("GET", url, **kwargs)

    def post(self, url: Any, **kwargs: Any):
        return self.request("POST", url, **kwargs)

    def put(self, url: Any, **kwargs: Any):
        return self.request("PUT", url, **kwargs)

    def patch(self, url: Any, **kwargs: Any):
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: Any, **kwargs: Any):
        return self.request("DELETE", url, **kwargs)

    def options(self, url: Any, **kwargs: Any):
        return self.request("OPTIONS", url, **kwargs)

    def head(self, url: Any, **kwargs: Any):
        return self.request("HEAD", url, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args: Any):
        return None
