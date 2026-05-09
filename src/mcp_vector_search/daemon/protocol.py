"""Wire protocol Pydantic models for the mvs daemon."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request sent from client to daemon."""

    type: str = Field(default="search")
    project_path: str
    query: str
    limit: int = 10
    mode: str = "hybrid"
    request_id: str = Field(default_factory=lambda: uuid4().hex)


class SearchResponse(BaseModel):
    """Search response sent from daemon to client."""

    request_id: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    project_path: str = ""
    error: str | None = None


class PingRequest(BaseModel):
    """Ping request used for liveness/status checks."""

    type: str = Field(default="ping")


class PingResponse(BaseModel):
    """Ping response with daemon metadata."""

    type: str = Field(default="pong")
    version: str
    uptime_s: float
    open_indexes: list[str] = Field(default_factory=list)


class ModelMismatchError(Exception):
    """Raised when a project's stored embedding model doesn't match the daemon."""

    def __init__(self, stored_model: str, loaded_model: str) -> None:
        self.stored_model = stored_model
        self.loaded_model = loaded_model
        super().__init__(
            f"Index built with {stored_model}, daemon loaded {loaded_model}. "
            f"Run 'mvs index --force' to rebuild."
        )
