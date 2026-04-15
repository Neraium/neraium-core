from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SecurityHeadersMiddleware:
    """Apply baseline browser security headers to every HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message.get("type") == "http.response.start":
                headers_list = list(message.get("headers") or [])
                existing = {k.lower() for k, _ in headers_list}

                def _set_if_absent(name: bytes, value: bytes) -> None:
                    if name.lower() not in existing:
                        headers_list.append((name, value))
                        existing.add(name.lower())

                _set_if_absent(b"x-content-type-options", b"nosniff")
                _set_if_absent(b"x-frame-options", b"DENY")
                _set_if_absent(b"referrer-policy", b"strict-origin-when-cross-origin")
                _set_if_absent(b"permissions-policy", b"camera=(), microphone=(), geolocation=()")
                _set_if_absent(b"cache-control", b"no-store")
                message["headers"] = headers_list
            await send(message)

        await self.app(scope, receive, send_with_security_headers)
