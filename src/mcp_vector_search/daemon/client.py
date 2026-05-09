"""Async client for the mvs daemon."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .paths import DAEMON_SOCK
from .protocol import PingResponse, SearchRequest, SearchResponse


class DaemonClient:
    """Connect to the daemon over a Unix socket and exchange JSON messages."""

    def __init__(self, sock_path: str | None = None) -> None:
        self.sock_path = sock_path or str(DAEMON_SOCK)

    @staticmethod
    def is_running() -> bool:
        """Return True if the daemon socket exists on disk."""
        return DAEMON_SOCK.exists()

    async def _send(
        self, payload: dict[str, Any], timeout_s: float
    ) -> dict[str, Any] | None:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.sock_path), timeout=timeout_s
            )
        except (TimeoutError, FileNotFoundError, ConnectionRefusedError, OSError):
            return None

        try:
            writer.write((json.dumps(payload) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
            if not line:
                return None
            return json.loads(line.decode("utf-8"))
        except (TimeoutError, json.JSONDecodeError, OSError):
            return None
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def search(
        self,
        project_path: str,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
        timeout_s: float = 10.0,
    ) -> SearchResponse | None:
        """Send a search request. Returns None when the daemon is unreachable."""
        req = SearchRequest(
            project_path=project_path, query=query, limit=limit, mode=mode
        )
        data = await self._send(req.model_dump(mode="json"), timeout_s)
        if data is None:
            return None
        return SearchResponse.model_validate(data)

    async def ping(self, timeout_s: float = 2.0) -> PingResponse | None:
        """Ping the daemon. Returns None when unreachable."""
        data = await self._send({"type": "ping"}, timeout_s)
        if data is None:
            return None
        try:
            return PingResponse.model_validate(data)
        except Exception:
            return None
