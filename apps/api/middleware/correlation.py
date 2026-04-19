from __future__ import annotations

from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestCorrelationIdMiddleware:
    """Ensure every request/response has a correlation ID for operator support."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        raw_id = headers.get(b"x-correlation-id") or headers.get(b"x-request-id")
        if raw_id:
            try:
                correlation_id = raw_id.decode("utf-8").strip()
            except UnicodeDecodeError:
                correlation_id = ""
        else:
            correlation_id = ""
        if not correlation_id:
            correlation_id = f"api_{uuid4().hex[:12]}"
        scope.setdefault("state", {})["correlation_id"] = correlation_id

        async def send_with_header(message: Message) -> None:
            if message.get("type") == "http.response.start":
                headers_list = list(message.get("headers") or [])
                headers_list.append((b"x-correlation-id", correlation_id.encode("utf-8")))
                message["headers"] = headers_list
            await send(message)

        await self.app(scope, receive, send_with_header)
