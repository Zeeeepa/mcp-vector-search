"""Asyncio Unix-socket daemon for persistent semantic search."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .. import __version__
from .paths import ensure_home
from .protocol import (
    ModelMismatchError,
    PingResponse,
    SearchRequest,
    SearchResponse,
)
from .registry import IndexRegistry

logger = logging.getLogger("mvs.daemon")


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(log_path), maxBytes=10 * 1024 * 1024, backupCount=3
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DaemonServer:
    """Single-host persistent daemon that serves search requests over a Unix socket."""

    def __init__(
        self,
        sock_path: Path,
        pid_path: Path,
        log_path: Path,
        max_indexes: int = 5,
        idle_timeout_s: float = 1800.0,
    ) -> None:
        self.sock_path = sock_path
        self.pid_path = pid_path
        self.log_path = log_path
        self.max_indexes = max_indexes
        self.idle_timeout_s = idle_timeout_s

        self._registry: IndexRegistry | None = None
        self._server: asyncio.base_events.Server | None = None
        self._start_time = time.monotonic()
        self._last_request_time = time.monotonic()
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Initialize the daemon and serve requests until stopped."""
        ensure_home()
        _setup_logging(self.log_path)
        logger.info(f"Daemon starting (version={__version__})")

        # Resolve loaded embedding model for mismatch checks. We use the
        # default model name from the project the daemon is launched against,
        # but since the daemon is project-agnostic we pull it from env or
        # leave it None (no check) until the first project is opened.
        loaded_model = os.environ.get("MVS_DAEMON_MODEL")
        self._registry = IndexRegistry(
            max_indexes=self.max_indexes, loaded_model=loaded_model
        )

        # Remove stale socket if present
        if self.sock_path.exists():
            try:
                self.sock_path.unlink()
            except OSError as e:  # pragma: no cover
                logger.error(f"Could not remove stale socket {self.sock_path}: {e}")

        # Write PID file
        self.pid_path.write_text(str(os.getpid()))

        # Start unix server
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(self.sock_path)
        )

        # Lock down socket permissions to user only
        try:
            os.chmod(self.sock_path, 0o600)
        except OSError as e:  # pragma: no cover
            logger.warning(f"Could not chmod socket: {e}")

        logger.info(f"Listening on {self.sock_path} (pid={os.getpid()})")

        # Signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._on_signal)
            except NotImplementedError:  # pragma: no cover
                pass

        watchdog = asyncio.create_task(self._idle_watchdog())

        try:
            await self._stop_event.wait()
        finally:
            watchdog.cancel()
            await self.stop()

    def _on_signal(self) -> None:
        logger.info("Received shutdown signal")
        self._stop_event.set()

    async def _idle_watchdog(self) -> None:
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(60)
                idle = time.monotonic() - self._last_request_time
                if idle > self.idle_timeout_s:
                    logger.info(f"Idle for {idle:.0f}s, shutting down")
                    self._stop_event.set()
                    return
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # pragma: no cover
                pass
            self._server = None
        if self._registry is not None:
            await self._registry.close_all()
        # Cleanup files
        for path in (self.sock_path, self.pid_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError:  # pragma: no cover
                pass
        logger.info("Daemon stopped")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                payload = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as e:
                resp = {"error": f"invalid JSON: {e}"}
                writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                await writer.drain()
                return

            kind = payload.get("type", "search")
            if kind == "ping":
                resp_obj: Any = await self._handle_ping()
            else:
                req = SearchRequest.model_validate(payload)
                resp_obj = await self._handle_search(req)

            data = resp_obj.model_dump(mode="json")
            writer.write((json.dumps(data) + "\n").encode("utf-8"))
            await writer.drain()
        except Exception as e:  # pragma: no cover - defensive
            logger.exception(f"Error handling client: {e}")
            try:
                err = json.dumps({"error": str(e)}).encode("utf-8") + b"\n"
                writer.write(err)
                await writer.drain()
            except Exception:
                pass
        finally:
            self._last_request_time = time.monotonic()
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # pragma: no cover
                pass

    async def _handle_ping(self) -> PingResponse:
        uptime = time.monotonic() - self._start_time
        open_indexes = self._registry.open_indexes() if self._registry else []
        return PingResponse(
            version=__version__,
            uptime_s=uptime,
            open_indexes=open_indexes,
        )

    async def _handle_search(self, req: SearchRequest) -> SearchResponse:
        if self._registry is None:
            return SearchResponse(
                request_id=req.request_id,
                project_path=req.project_path,
                error="Daemon registry not initialised",
            )
        t0 = time.monotonic()
        try:
            engine, lock = await self._registry.get_or_open(req.project_path)
        except ModelMismatchError as e:
            return SearchResponse(
                request_id=req.request_id,
                project_path=req.project_path,
                error=str(e),
            )
        except FileNotFoundError as e:
            return SearchResponse(
                request_id=req.request_id,
                project_path=req.project_path,
                error=f"Project index not found: {e}",
            )
        except Exception as e:
            logger.exception(f"Failed to open project {req.project_path}")
            return SearchResponse(
                request_id=req.request_id,
                project_path=req.project_path,
                error=f"Failed to open project: {e}",
            )

        from ..core.search import SearchMode

        try:
            mode = SearchMode(req.mode.lower())
        except ValueError:
            mode = SearchMode.HYBRID

        async with lock:
            try:
                results = await engine.search(
                    query=req.query,
                    limit=req.limit,
                    search_mode=mode,
                )
            except Exception as e:
                logger.exception(f"Search failed for {req.project_path}")
                return SearchResponse(
                    request_id=req.request_id,
                    project_path=req.project_path,
                    error=f"Search failed: {e}",
                )

        latency = (time.monotonic() - t0) * 1000.0
        return SearchResponse(
            request_id=req.request_id,
            project_path=req.project_path,
            results=[r.to_dict() for r in results],
            latency_ms=latency,
        )
