"""Entry point for `python -m mcp_vector_search.daemon`."""

from __future__ import annotations

import asyncio

from .paths import DAEMON_LOG, DAEMON_PID, DAEMON_SOCK
from .server import DaemonServer


async def _amain() -> None:
    server = DaemonServer(
        sock_path=DAEMON_SOCK,
        pid_path=DAEMON_PID,
        log_path=DAEMON_LOG,
    )
    await server.start()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
