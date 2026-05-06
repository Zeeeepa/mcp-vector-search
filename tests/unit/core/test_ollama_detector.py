"""Unit tests for mcp_vector_search.core.ollama_detector.

Covers:
- Context limit detection by model size substring
- Ollama detection (success / failure / non-200)
- Model listing (success / failure / empty / malformed)
- Best-model selection preference ordering and fallback
- Sync wrappers
- All HTTP calls are mocked — no real network traffic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mcp_vector_search.core import ollama_detector
from mcp_vector_search.core.ollama_detector import (
    CONTEXT_LIMITS,
    PREFERRED_MODELS,
    detect_best_model,
    detect_best_model_sync,
    detect_ollama,
    detect_ollama_sync,
    get_context_limits,
    get_model_limits,
    list_ollama_models,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_async_client(get_return=None, get_side_effect=None):
    """Build a patcher for httpx.AsyncClient whose ctx-manager .get() is mocked.

    Either `get_return` (the awaited return value of .get) or `get_side_effect`
    (an exception to raise from .get) must be provided.
    """
    client_instance = MagicMock()
    if get_side_effect is not None:
        client_instance.get = AsyncMock(side_effect=get_side_effect)
    else:
        client_instance.get = AsyncMock(return_value=get_return)

    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=client_instance)
    async_cm.__aexit__ = AsyncMock(return_value=None)

    return patch.object(ollama_detector.httpx, "AsyncClient", return_value=async_cm)


def _mock_response(status_code=200, json_data=None):
    """Build a mock httpx Response with given status and json payload."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_data if json_data is not None else {})
    return resp


# ---------------------------------------------------------------------------
# Tests: get_context_limits / get_model_limits
# ---------------------------------------------------------------------------


class TestGetContextLimits:
    """Tests for get_context_limits()."""

    def test_returns_default_for_unknown_model(self):
        """Unknown model size falls back to default 8/5."""
        result = get_context_limits("totally-made-up-model:xyz")
        assert result == {"max_iterations": 8, "max_results": 5}

    def test_matches_405b(self):
        result = get_context_limits("llama3.1:405b")
        assert result == {"max_iterations": 20, "max_results": 10}

    def test_matches_70b(self):
        result = get_context_limits("llama3.1:70b")
        assert result == {"max_iterations": 15, "max_results": 8}

    def test_matches_72b(self):
        result = get_context_limits("qwen2.5:72b")
        assert result == {"max_iterations": 15, "max_results": 8}

    def test_matches_27b(self):
        result = get_context_limits("gemma2:27b")
        assert result == {"max_iterations": 10, "max_results": 6}

    def test_matches_7b_in_compound_tag(self):
        """qwen2.5-coder:7b-instruct should still match the '7b' bucket."""
        result = get_context_limits("qwen2.5-coder:7b-instruct")
        assert result == {"max_iterations": 8, "max_results": 5}

    def test_matches_2b(self):
        result = get_context_limits("gemma2:2b")
        assert result == {"max_iterations": 3, "max_results": 3}

    def test_matches_e4b(self):
        """e4b is a distinct bucket — must not collide with 4b."""
        result = get_context_limits("gemma4:e4b")
        assert result == {"max_iterations": 8, "max_results": 5}

    def test_case_insensitive(self):
        """Matching is performed on the lowercased model name."""
        result = get_context_limits("LLAMA3.1:70B")
        assert result == {"max_iterations": 15, "max_results": 8}

    def test_no_size_token_returns_default(self):
        """A model name without any size token returns default limits."""
        result = get_context_limits("custom-model:latest")
        assert result == {"max_iterations": 8, "max_results": 5}

    def test_first_match_wins_ordering(self):
        """The dict is ordered largest-first; pick the first matching key."""
        # 'phi3' contains '3' but no key in CONTEXT_LIMITS is just '3', so default.
        # However a name with both '70b' and '7b' should match '70b' first.
        result = get_context_limits("frankenmodel:70b-7b-mix")
        assert result == {"max_iterations": 15, "max_results": 8}

    def test_get_model_limits_is_alias(self):
        """get_model_limits delegates to get_context_limits."""
        for name in ("qwen2.5:7b", "llama3.1:70b", "unknown:tag"):
            assert get_model_limits(name) == get_context_limits(name)

    def test_all_context_limits_keys_have_required_fields(self):
        """Each entry in CONTEXT_LIMITS exposes both required keys."""
        for size_key, limits in CONTEXT_LIMITS.items():
            assert "max_iterations" in limits, size_key
            assert "max_results" in limits, size_key
            assert isinstance(limits["max_iterations"], int)
            assert isinstance(limits["max_results"], int)


# ---------------------------------------------------------------------------
# Tests: detect_ollama
# ---------------------------------------------------------------------------


class TestDetectOllama:
    """Tests for detect_ollama()."""

    @pytest.mark.asyncio
    async def test_returns_true_on_200(self):
        with _mock_async_client(get_return=_mock_response(200, {})):
            assert await detect_ollama() is True

    @pytest.mark.asyncio
    async def test_returns_false_on_non_200(self):
        with _mock_async_client(get_return=_mock_response(500, {})):
            assert await detect_ollama() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_refused(self):
        """Ollama not running -> ConnectError -> returns False, no crash."""
        with _mock_async_client(
            get_side_effect=httpx.ConnectError("connection refused")
        ):
            assert await detect_ollama() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        with _mock_async_client(get_side_effect=httpx.ReadTimeout("timed out")):
            assert await detect_ollama() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_unexpected_exception(self):
        """Any Exception subtype is swallowed -> False."""
        with _mock_async_client(get_side_effect=RuntimeError("boom")):
            assert await detect_ollama() is False


# ---------------------------------------------------------------------------
# Tests: list_ollama_models
# ---------------------------------------------------------------------------


class TestListOllamaModels:
    """Tests for list_ollama_models()."""

    @pytest.mark.asyncio
    async def test_returns_model_names(self):
        payload = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "qwen2.5-coder:7b"},
            ]
        }
        with _mock_async_client(get_return=_mock_response(200, payload)):
            result = await list_ollama_models()
        assert result == ["llama3.1:8b", "qwen2.5-coder:7b"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_models_field(self):
        with _mock_async_client(get_return=_mock_response(200, {})):
            assert await list_ollama_models() == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_models_list_empty(self):
        with _mock_async_client(get_return=_mock_response(200, {"models": []})):
            assert await list_ollama_models() == []

    @pytest.mark.asyncio
    async def test_skips_entries_missing_name(self):
        """Entries without a 'name' key are silently skipped."""
        payload = {
            "models": [
                {"name": "llama3.1:8b"},
                {"size": 1234},  # no name key
                {"name": "phi4:latest"},
            ]
        }
        with _mock_async_client(get_return=_mock_response(200, payload)):
            result = await list_ollama_models()
        assert result == ["llama3.1:8b", "phi4:latest"]

    @pytest.mark.asyncio
    async def test_returns_empty_on_non_200(self):
        with _mock_async_client(get_return=_mock_response(503, {})):
            assert await list_ollama_models() == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_connection_error(self):
        """Connection refused -> empty list (Ollama not running)."""
        with _mock_async_client(
            get_side_effect=httpx.ConnectError("connection refused")
        ):
            assert await list_ollama_models() == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self):
        """If .json() raises, function swallows and returns []."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(side_effect=ValueError("invalid json"))
        with _mock_async_client(get_return=resp):
            assert await list_ollama_models() == []


# ---------------------------------------------------------------------------
# Tests: detect_best_model
# ---------------------------------------------------------------------------


class TestDetectBestModel:
    """Tests for detect_best_model() preference ordering and fallback."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_models(self):
        """Empty model list (Ollama not running) -> None."""
        with patch.object(
            ollama_detector,
            "list_ollama_models",
            new=AsyncMock(return_value=[]),
        ):
            assert await detect_best_model() is None

    @pytest.mark.asyncio
    async def test_returns_none_when_ollama_unreachable(self):
        """list_ollama_models returns [] when unreachable -> detect returns None."""
        with _mock_async_client(
            get_side_effect=httpx.ConnectError("connection refused")
        ):
            assert await detect_best_model() is None

    @pytest.mark.asyncio
    async def test_picks_highest_preference_when_multiple_available(self):
        """When several preferred models exist, pick the earliest in PREFERRED_MODELS."""
        # gemma3:latest is earlier in PREFERRED_MODELS than llama3.1:8b.
        available = ["llama3.1:8b", "phi4:latest", "gemma3:latest", "mistral:latest"]
        with patch.object(
            ollama_detector,
            "list_ollama_models",
            new=AsyncMock(return_value=available),
        ):
            assert await detect_best_model() == "gemma3:latest"

    @pytest.mark.asyncio
    async def test_substring_match_variant(self):
        """A variant tag like qwen2.5-coder:7b-instruct still matches 7b prefs."""
        available = ["qwen2.5-coder:7b-instruct"]
        with patch.object(
            ollama_detector,
            "list_ollama_models",
            new=AsyncMock(return_value=available),
        ):
            result = await detect_best_model()
        assert result == "qwen2.5-coder:7b-instruct"

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self):
        """Matching is case-insensitive in both directions."""
        available = ["LLAMA3.1:8B"]
        with patch.object(
            ollama_detector,
            "list_ollama_models",
            new=AsyncMock(return_value=available),
        ):
            # Should still return the original-cased name, matched via 'llama3.1:8b' pref.
            assert await detect_best_model() == "LLAMA3.1:8B"

    @pytest.mark.asyncio
    async def test_falls_back_to_first_available_when_none_preferred(self):
        """No preference matches -> return the first installed model."""
        available = ["custom-model-xyz:latest", "another-custom:latest"]
        with patch.object(
            ollama_detector,
            "list_ollama_models",
            new=AsyncMock(return_value=available),
        ):
            assert await detect_best_model() == "custom-model-xyz:latest"

    @pytest.mark.asyncio
    async def test_single_preferred_model(self):
        available = ["phi4:latest"]
        with patch.object(
            ollama_detector,
            "list_ollama_models",
            new=AsyncMock(return_value=available),
        ):
            assert await detect_best_model() == "phi4:latest"

    @pytest.mark.asyncio
    async def test_preference_ordering_gemma_before_llama(self):
        """Gemma family has higher priority than Llama family."""
        # Confirm assumption holds in the constant.
        gemma_idx = next(
            i for i, m in enumerate(PREFERRED_MODELS) if m.startswith("gemma")
        )
        llama_idx = next(
            i for i, m in enumerate(PREFERRED_MODELS) if m.startswith("llama")
        )
        assert gemma_idx < llama_idx

        available = ["llama3.1:70b", "gemma2:9b"]
        with patch.object(
            ollama_detector,
            "list_ollama_models",
            new=AsyncMock(return_value=available),
        ):
            assert await detect_best_model() == "gemma2:9b"


# ---------------------------------------------------------------------------
# Tests: sync wrappers
# ---------------------------------------------------------------------------


class TestSyncWrappers:
    """Tests for detect_ollama_sync() and detect_best_model_sync()."""

    def test_detect_ollama_sync_true(self):
        with patch.object(
            ollama_detector,
            "detect_ollama",
            new=AsyncMock(return_value=True),
        ):
            assert detect_ollama_sync() is True

    def test_detect_ollama_sync_false(self):
        with patch.object(
            ollama_detector,
            "detect_ollama",
            new=AsyncMock(return_value=False),
        ):
            assert detect_ollama_sync() is False

    def test_detect_best_model_sync_returns_model(self):
        with patch.object(
            ollama_detector,
            "detect_best_model",
            new=AsyncMock(return_value="phi4:latest"),
        ):
            assert detect_best_model_sync() == "phi4:latest"

    def test_detect_best_model_sync_returns_none(self):
        with patch.object(
            ollama_detector,
            "detect_best_model",
            new=AsyncMock(return_value=None),
        ):
            assert detect_best_model_sync() is None

    def test_detect_ollama_sync_inside_running_loop(self):
        """When asyncio.run raises RuntimeError, falls back to thread executor."""
        # Simulate "already inside event loop" by making asyncio.run raise.
        call_count = {"n": 0}

        def fake_run(coro):
            call_count["n"] += 1
            # First call (top-level) raises; second call (inside thread) succeeds.
            if call_count["n"] == 1:
                # Close the coroutine to avoid "coroutine never awaited" warnings.
                coro.close()
                raise RuntimeError("asyncio.run() cannot be called from a running loop")
            coro.close()
            return True

        with patch.object(ollama_detector.asyncio, "run", side_effect=fake_run):
            assert detect_ollama_sync() is True
        assert call_count["n"] == 2

    def test_detect_best_model_sync_inside_running_loop(self):
        """Sync best-model wrapper falls back to thread executor on RuntimeError."""
        call_count = {"n": 0}

        def fake_run(coro):
            call_count["n"] += 1
            if call_count["n"] == 1:
                coro.close()
                raise RuntimeError("asyncio.run() cannot be called from a running loop")
            coro.close()
            return "gemma3:latest"

        with patch.object(ollama_detector.asyncio, "run", side_effect=fake_run):
            assert detect_best_model_sync() == "gemma3:latest"
        assert call_count["n"] == 2
