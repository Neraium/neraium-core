from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_SECURITY_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"x-permitted-cross-domain-policies", b"none"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]


class SecurityHeadersMiddleware:
    """Attach security response headers to every HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        scheme = str(scope.get("scheme") or "").lower()

        async def send_with_headers(message: Message) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                existing_names = {name.lower() for name, _ in headers}
                for name, value in _SECURITY_HEADERS:
                    if name not in existing_names:
                        headers.append((name, value))
                # Prevent caching of JSON API responses (especially /health).
                if b"cache-control" not in existing_names:
                    content_type = next(
                        (value for name, value in headers if name.lower() == b"content-type"),
                        b"",
                    )
                    if b"application/json" in content_type.lower():
                        headers.append((b"cache-control", b"no-store"))
                # Only attach HSTS on HTTPS responses.
                if scheme == "https" and b"strict-transport-security" not in existing_names:
                    headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)
