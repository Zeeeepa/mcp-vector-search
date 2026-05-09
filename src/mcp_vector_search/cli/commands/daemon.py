"""`mvs daemon` Typer subcommands."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Persistent search daemon")


def _read_pid() -> int | None:
    from ...daemon.paths import DAEMON_PID

    if not DAEMON_PID.exists():
        return None
    try:
        return int(DAEMON_PID.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@app.command("start")
def start() -> None:
    """Start the daemon if it is not already running."""
    from ...daemon.client import DaemonClient
    from ...daemon.paths import DAEMON_SOCK, ensure_home

    if DaemonClient.is_running():
        console.print("[yellow]Daemon already running[/yellow]")
        raise typer.Exit(0)

    ensure_home()
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_vector_search.daemon"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for socket to appear
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if DAEMON_SOCK.exists():
            console.print(f"[green]✓[/green] Daemon started (PID {proc.pid})")
            raise typer.Exit(0)
        time.sleep(0.1)

    console.print("[red]✗ Daemon failed to start within 5s[/red]")
    raise typer.Exit(1)


@app.command("stop")
def stop() -> None:
    """Stop the running daemon."""
    from ...daemon.paths import DAEMON_SOCK

    pid = _read_pid()
    if pid is None or not _is_alive(pid):
        console.print("[yellow]Daemon not running[/yellow]")
        raise typer.Exit(0)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        console.print("[yellow]Daemon already stopped[/yellow]")
        raise typer.Exit(0)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not DAEMON_SOCK.exists():
            console.print("[green]✓[/green] Daemon stopped")
            raise typer.Exit(0)
        time.sleep(0.1)

    console.print("[yellow]Daemon did not exit cleanly within 5s[/yellow]")
    raise typer.Exit(1)


@app.command("status")
def status() -> None:
    """Show daemon status and open indexes."""
    from ...daemon.client import DaemonClient

    async def _run() -> int:
        client = DaemonClient()
        resp = await client.ping()
        if resp is None:
            console.print("[yellow]Daemon: not running[/yellow]")
            return 1
        console.print(f"[green]Daemon: running[/green] (version {resp.version})")
        console.print(f"  uptime: {resp.uptime_s:.1f}s")
        console.print(f"  open indexes: {len(resp.open_indexes)}")
        for path in resp.open_indexes:
            console.print(f"    • {path}")
        return 0

    code = asyncio.run(_run())
    raise typer.Exit(code)


@app.command("restart")
def restart() -> None:
    """Stop and then start the daemon."""
    try:
        stop()
    except typer.Exit:
        pass
    # Give the OS a moment to release the socket file
    time.sleep(0.5)
    start()
