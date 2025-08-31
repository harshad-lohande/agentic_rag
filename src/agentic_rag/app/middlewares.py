# src/agentic_rag/app/middlewares.py

import uuid
from contextvars import ContextVar
from typing import Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# A context variable to hold the request ID for the duration of a request.
REQUEST_ID_CTX_VAR: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique X-Request-ID header into every request.
    This ID is then available throughout the application via a context variable.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Set the context variable
        REQUEST_ID_CTX_VAR.set(request_id)

        response = await call_next(request)

        # Also add the ID to the response headers
        response.headers["X-Request-ID"] = request_id
        return response