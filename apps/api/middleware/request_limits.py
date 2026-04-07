from __future__ import annotations

from fastapi import status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyTooLargeError(Exception):
    """Raised when an incoming request body exceeds configured max size."""


class MaxRequestBodySizeMiddleware:
    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        self.app = app
        self.max_body_size = max(1, int(max_body_size))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        content_type = (headers.get(b"content-type") or b"").decode("latin1", errors="ignore").lower()
        if "multipart/form-data" in content_type:
            await self.app(scope, receive, send)
            return
        raw_content_length = headers.get(b"content-length")
        if raw_content_length:
            try:
                content_length = int(raw_content_length.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                content_length = None
            if content_length is not None and content_length > self.max_body_size:
                await self._send_413(scope, receive, send)
                return

        bytes_seen = 0
        response_started = False

        async def guarded_receive() -> Message:
            nonlocal bytes_seen
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                bytes_seen += len(body)
                if bytes_seen > self.max_body_size:
                    raise RequestBodyTooLargeError
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, tracked_send)
        except RequestBodyTooLargeError:
            if not response_started:
                await self._send_413(scope, receive, send)

    async def _send_413(self, scope: Scope, receive: Receive, send: Send) -> None:
        max_mb = self.max_body_size / (1024 * 1024)
        response = JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"detail": f"Request body too large (max {max_mb:.1f}MB)."},
        )
        await response(scope, receive, send)
