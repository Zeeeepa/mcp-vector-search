"""Detect Ollama and available local models."""

import asyncio

import httpx
from loguru import logger

OLLAMA_BASE_URL = "http://localhost:11434"

# Ordered preference list — first match wins (case-insensitive substring match)
PREFERRED_MODELS = [
    # Gemma3 - best for code, supports function calling
    "gemma3:latest",
    "gemma3:27b",
    "gemma3:12b",
    "gemma3:9b",
    "gemma3:4b",
    "gemma3:2b",
    # Gemma2
    "gemma2:latest",
    "gemma2:27b",
    "gemma2:9b",
    "gemma2:2b",
    # Qwen2.5-coder - excellent code reasoning, function calling
    "qwen2.5-coder:latest",
    "qwen2.5-coder:32b",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:7b-instruct",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:3b",
    # Qwen2.5 general
    "qwen2.5:latest",
    "qwen2.5:72b",
    "qwen2.5:32b",
    "qwen2.5:14b",
    "qwen2.5:7b",
    # DeepSeek coder
    "deepseek-coder-v2:latest",
    "deepseek-v3.1:latest",
    "deepseek-v3:latest",
    # Codellama
    "codellama:latest",
    "codellama:70b",
    "codellama:34b",
    "codellama:13b",
    "codellama:7b",
    # Llama3 - good all-rounder, function calling support
    "llama3.1:latest",
    "llama3.1:405b",
    "llama3.1:70b",
    "llama3.1:8b",
    "llama3.2:latest",
    "llama3.2:3b",
    "llama3:latest",
    "llama3:70b",
    # Phi
    "phi4:latest",
    "phi4-mini:latest",
    "phi3:latest",
    # Mistral - last resort (poor function calling format compliance)
    "mistral-small3.2:latest",
    "mistral:latest",
    # IQuest/other HF models
    "hf.co/ilintar/IQuest-Coder-V1-40B-Instruct-GGUF:latest",
]

# Iteration/result limits keyed by model size substring
CONTEXT_LIMITS: dict[str, dict[str, int]] = {
    "405b": {"max_iterations": 20, "max_results": 10},
    "70b": {"max_iterations": 15, "max_results": 8},
    "72b": {"max_iterations": 15, "max_results": 8},
    "40b": {"max_iterations": 12, "max_results": 7},
    "32b": {"max_iterations": 12, "max_results": 7},
    "27b": {"max_iterations": 10, "max_results": 6},
    "14b": {"max_iterations": 10, "max_results": 6},
    "12b": {"max_iterations": 10, "max_results": 6},
    "9b": {"max_iterations": 8, "max_results": 5},
    "7b": {"max_iterations": 8, "max_results": 5},
    "4b": {"max_iterations": 5, "max_results": 3},
    "3b": {"max_iterations": 5, "max_results": 3},
    "2b": {"max_iterations": 3, "max_results": 3},
}


def get_context_limits(model_name: str) -> dict[str, int]:
    """Return iteration/result limits based on model size.

    Args:
        model_name: Full model name e.g. 'qwen2.5-coder:7b-instruct'

    Returns:
        Dict with 'max_iterations' and 'max_results' keys
    """
    model_lower = model_name.lower()
    for size_key, limits in CONTEXT_LIMITS.items():
        if size_key in model_lower:
            return limits
    return {"max_iterations": 8, "max_results": 5}  # default for unknown sizes


def get_model_limits(model_name: str) -> dict[str, int]:
    """Alias for get_context_limits — kept for backward compatibility.

    Args:
        model_name: Full model name e.g. 'qwen2.5-coder:7b-instruct'

    Returns:
        Dict with 'max_iterations' and 'max_results' keys
    """
    return get_context_limits(model_name)


async def detect_ollama() -> bool:
    """Check if Ollama is running by hitting the /api/tags endpoint.

    Returns:
        True if Ollama is reachable, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except Exception as exc:
        logger.debug(f"Ollama not detected: {exc}")
        return False


async def list_ollama_models() -> list[str]:
    """List available models via GET /api/tags.

    Returns:
        List of model name strings (may be empty if Ollama unreachable)
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = data.get("models", [])
            return [m["name"] for m in models if "name" in m]
    except Exception as exc:
        logger.debug(f"Failed to list Ollama models: {exc}")
        return []


async def detect_best_model() -> str | None:
    """Pick the best available local model.

    Preference order is defined by PREFERRED_MODELS.  Matching is
    case-insensitive substring: an available model is selected if its
    lowercased name contains the lowercased preference string (or vice
    versa).  This handles variants like 'qwen2.5-coder:7b-instruct'
    matching the preference 'qwen2.5-coder:7b-instruct'.

    Returns the first available model if none of the preferred models
    are installed, or None if Ollama is not running.

    Returns:
        Model name string, or None
    """
    available = await list_ollama_models()
    if not available:
        return None

    available_lower = [(m.lower(), m) for m in available]

    # Try preferred list first (case-insensitive substring match in both directions)
    for preferred in PREFERRED_MODELS:
        preferred_lower = preferred.lower()
        for avail_lower, avail_orig in available_lower:
            if (
                preferred_lower == avail_lower
                or preferred_lower in avail_lower
                or avail_lower in preferred_lower
            ):
                return avail_orig

    # Fall back to whatever is installed
    return available[0]


def detect_ollama_sync() -> bool:
    """Synchronous wrapper around detect_ollama().

    Returns:
        True if Ollama is reachable, False otherwise
    """
    try:
        return asyncio.run(detect_ollama())
    except RuntimeError:
        # Already inside an event loop — use a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, detect_ollama())
            return future.result(timeout=5)


def detect_best_model_sync() -> str | None:
    """Synchronous wrapper around detect_best_model().

    Returns:
        Model name string, or None
    """
    try:
        return asyncio.run(detect_best_model())
    except RuntimeError:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, detect_best_model())
            return future.result(timeout=5)
