"""Canonical paths for the mvs daemon."""

from __future__ import annotations

import os
from pathlib import Path

MVS_HOME = Path(os.environ.get("MVS_HOME", str(Path.home() / ".mcp-vector-search")))
DAEMON_SOCK = MVS_HOME / "daemon.sock"
DAEMON_PID = MVS_HOME / "daemon.pid"
DAEMON_LOG = MVS_HOME / "daemon.log"


def ensure_home() -> Path:
    """Ensure the MVS_HOME directory exists. Returns the resolved path."""
    MVS_HOME.mkdir(parents=True, exist_ok=True)
    return MVS_HOME
