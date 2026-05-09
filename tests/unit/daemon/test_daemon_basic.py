"""Basic unit tests for the mvs daemon (registry + client)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mcp_vector_search.daemon.client import DaemonClient
from mcp_vector_search.daemon.protocol import (
    ModelMismatchError,
    SearchRequest,
    SearchResponse,
)
from mcp_vector_search.daemon.registry import IndexRegistry


def test_search_request_defaults_request_id() -> None:
    req = SearchRequest(project_path="/tmp", query="hello")
    assert req.request_id  # auto-generated uuid hex
    assert req.limit == 10
    assert req.mode == "hybrid"


def test_search_response_round_trip() -> None:
    resp = SearchResponse(
        request_id="abc",
        results=[{"file_path": "a.py"}],
        latency_ms=12.5,
        project_path="/x",
    )
    blob = resp.model_dump_json()
    parsed = SearchResponse.model_validate(json.loads(blob))
    assert parsed.request_id == "abc"
    assert parsed.latency_ms == 12.5


def test_registry_model_mismatch(tmp_path: Path) -> None:
    """Registry must raise ModelMismatchError when stored model differs."""
    project = tmp_path / "proj"
    (project / ".mcp-vector-search").mkdir(parents=True)
    meta = project / ".mcp-vector-search" / "index_metadata.json"
    meta.write_text(json.dumps({"embedding_model": "model-A"}))

    registry = IndexRegistry(loaded_model="model-B")
    with pytest.raises(ModelMismatchError) as exc_info:
        registry._check_model(project)
    assert "model-A" in str(exc_info.value)
    assert "model-B" in str(exc_info.value)


def test_registry_model_match_noop(tmp_path: Path) -> None:
    """No exception when stored model matches loaded model."""
    project = tmp_path / "proj"
    (project / ".mcp-vector-search").mkdir(parents=True)
    meta = project / ".mcp-vector-search" / "index_metadata.json"
    meta.write_text(json.dumps({"embedding_model": "same-model"}))

    registry = IndexRegistry(loaded_model="same-model")
    registry._check_model(project)  # should not raise


def test_registry_no_metadata_no_check(tmp_path: Path) -> None:
    """Missing metadata file is treated as 'trust the daemon'."""
    project = tmp_path / "proj"
    project.mkdir()
    registry = IndexRegistry(loaded_model="any-model")
    registry._check_model(project)  # should not raise


def test_daemon_client_returns_none_when_no_socket(tmp_path: Path) -> None:
    """Client should return None when the socket does not exist."""
    fake_sock = tmp_path / "missing.sock"
    client = DaemonClient(sock_path=str(fake_sock))

    async def _run() -> object:
        return await client.search(project_path=str(tmp_path), query="x", timeout_s=0.5)

    result = asyncio.run(_run())
    assert result is None


def test_daemon_client_is_running_false_when_default_sock_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """is_running() reflects existence of the canonical daemon socket."""
    monkeypatch.setattr(
        "mcp_vector_search.daemon.client.DAEMON_SOCK", tmp_path / "absent.sock"
    )
    assert DaemonClient.is_running() is False
