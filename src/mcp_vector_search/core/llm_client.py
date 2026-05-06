"""LLM client for intelligent code search using OpenAI, OpenRouter, AWS Bedrock, or Ollama API."""

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from loguru import logger

from .exceptions import SearchError

# Type alias for provider
LLMProvider = Literal["openai", "openrouter", "bedrock", "ollama"]

# Type alias for intent
IntentType = Literal["find", "answer", "analyze"]

# XML tag used for prompt-engineered tool calls (Ollama/local models)
_TOOL_CALL_TAG = "tool_call"
_TOOL_CALL_OPEN = f"<{_TOOL_CALL_TAG}>"
_TOOL_CALL_CLOSE = f"</{_TOOL_CALL_TAG}>"


class LLMClient:
    """Client for LLM-powered intelligent search orchestration.

    Supports OpenAI, OpenRouter, and AWS Bedrock APIs:
    1. Generate multiple targeted search queries from natural language
    2. Analyze search results and select most relevant ones
    3. Provide contextual explanations for results

    Provider Selection Priority:
    1. Explicit provider parameter
    2. Preferred provider from config
    3. Auto-detect: Bedrock (if AWS creds) → OpenRouter → OpenAI

    Default Models:
    - Bedrock: Claude 3.5 Haiku (anthropic.claude-3-5-haiku-20241022-v1:0)
    - OpenRouter: Claude Opus 4.5
    - OpenAI: GPT-4o-mini
    """

    # Default models for each provider (comparable performance/cost)
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",  # Fast, cheap, comparable to claude-3-haiku
        "openrouter": "anthropic/claude-opus-4.5",  # Claude Opus 4.5 for chat REPL
        "bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0",  # Claude 3.5 Haiku (valid cross-region profile)
        "ollama": "gemma3:latest",  # Best open-weights model for local inference
    }

    # Advanced "thinking" models for complex queries (--think flag)
    THINKING_MODELS = {
        "openai": "gpt-4o",  # More capable, better reasoning
        "openrouter": "anthropic/claude-opus-4.5",  # Claude Opus 4.5 for deep analysis
        "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Claude 3.5 Sonnet v2 (valid cross-region profile)
    }

    # API endpoints
    API_ENDPOINTS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "ollama": "http://localhost:11434/v1/chat/completions",  # OpenAI-compatible
    }

    TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
        provider: LLMProvider | None = None,
        openai_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        think: bool = False,
    ) -> None:
        """Initialize LLM client.

        Args:
            api_key: API key (deprecated, use provider-specific keys)
            model: Model to use (defaults based on provider)
            timeout: Request timeout in seconds
            provider: Explicit provider ('openai', 'openrouter', or 'bedrock')
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            openrouter_api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
            think: Use advanced "thinking" model for complex queries

        Raises:
            ValueError: If no API key/credentials found for any provider
        """
        self.think = think
        self._bedrock_client = None  # Lazy initialization

        # Get API keys from environment or parameters
        self.openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

        # Support deprecated api_key parameter (assume OpenRouter for backward compatibility)
        if api_key and not self.openrouter_key:
            self.openrouter_key = api_key

        # Determine which provider to use
        self.provider: LLMProvider = self._resolve_provider(provider)

        # Set API key, endpoint, and model based on provider
        self._configure_provider(model, think)

        self.timeout = timeout

        logger.debug(
            f"Initialized LLM client with provider: {self.provider}, model: {self.model}"
        )

    def _resolve_provider(self, provider: LLMProvider | None) -> LLMProvider:
        """Resolve which provider to use based on explicit choice or auto-detect.

        Args:
            provider: Explicit provider, or None for auto-detect

        Returns:
            Resolved provider name

        Raises:
            ValueError: If credentials are missing for the chosen/detected provider
        """
        if provider:
            self._validate_provider_credentials(provider)
            return provider

        # Auto-detect provider (prefer Bedrock → OpenRouter → OpenAI)
        if self._bedrock_available:
            return "bedrock"
        if self.openrouter_key:
            return "openrouter"
        if self.openai_key:
            return "openai"
        raise ValueError(
            "No API key or AWS credentials found. Please set AWS credentials "
            "(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) for Bedrock, "
            "OPENROUTER_API_KEY for OpenRouter, or OPENAI_API_KEY for OpenAI. "
            "Alternatively, start Ollama (ollama serve) and use --provider ollama."
        )

    def _validate_provider_credentials(self, provider: LLMProvider) -> None:
        """Validate that credentials exist for an explicitly-specified provider.

        Args:
            provider: Provider whose credentials should be checked

        Raises:
            ValueError: If credentials for the provider are missing
        """
        if provider == "openai" and not self.openai_key:
            raise ValueError(
                "OpenAI provider specified but OPENAI_API_KEY not found. "
                "Please set OPENAI_API_KEY environment variable."
            )
        if provider == "openrouter" and not self.openrouter_key:
            raise ValueError(
                "OpenRouter provider specified but OPENROUTER_API_KEY not found. "
                "Please set OPENROUTER_API_KEY environment variable."
            )
        if provider == "bedrock" and not self._bedrock_available:
            raise ValueError(
                "Bedrock provider specified but AWS credentials not found. "
                "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )
        # ollama: no key required — validated at connection time

    def _configure_provider(self, model: str | None, think: bool) -> None:
        """Configure api_key, api_endpoint, and model attributes for current provider.

        Selection precedence: explicit model > env var > thinking model > default model

        Args:
            model: Explicit model override (or None to use env/default)
            think: Whether to prefer thinking-tier models
        """
        if self.provider == "openai":
            self._configure_openai(model, think)
        elif self.provider == "openrouter":
            self._configure_openrouter(model, think)
        elif self.provider == "ollama":
            self._configure_ollama(model)
        else:  # bedrock
            self._configure_bedrock(model, think)

    def _configure_openai(self, model: str | None, think: bool) -> None:
        """Configure attributes for the OpenAI provider."""
        self.api_key = self.openai_key
        self.api_endpoint = self.API_ENDPOINTS["openai"]
        default_model = (
            self.THINKING_MODELS["openai"] if think else self.DEFAULT_MODELS["openai"]
        )
        self.model = model or os.environ.get("OPENAI_MODEL", default_model)

    def _configure_openrouter(self, model: str | None, think: bool) -> None:
        """Configure attributes for the OpenRouter provider."""
        self.api_key = self.openrouter_key
        self.api_endpoint = self.API_ENDPOINTS["openrouter"]
        default_model = (
            self.THINKING_MODELS["openrouter"]
            if think
            else self.DEFAULT_MODELS["openrouter"]
        )
        self.model = model or os.environ.get("OPENROUTER_MODEL", default_model)

    def _configure_ollama(self, model: str | None) -> None:
        """Configure attributes for the Ollama provider (no auth required)."""
        self.api_key = None  # No auth required for local Ollama
        self.api_endpoint = os.environ.get(
            "OLLAMA_API_URL", self.API_ENDPOINTS["ollama"]
        )
        self.model = model or os.environ.get(
            "OLLAMA_MODEL", self.DEFAULT_MODELS["ollama"]
        )

    def _configure_bedrock(self, model: str | None, think: bool) -> None:
        """Configure attributes for the Bedrock provider (no HTTP endpoint)."""
        self.api_key = None  # Not used for Bedrock
        self.api_endpoint = None  # Not used for Bedrock
        default_model = (
            self.THINKING_MODELS["bedrock"] if think else self.DEFAULT_MODELS["bedrock"]
        )
        self.model = model or os.environ.get("BEDROCK_MODEL", default_model)

    def _get_endpoint(self) -> str:
        """Get API endpoint URL for current provider, raising if not configured.

        Returns:
            The API endpoint URL string

        Raises:
            ValueError: If endpoint is not configured for the provider
        """
        endpoint = self.api_endpoint  # local var so Pyright can narrow str | None → str
        if endpoint is None:
            raise ValueError(
                f"No API endpoint configured for provider: {self.provider}. "
                "This is typically only used for Bedrock, which doesn't use HTTP endpoints."
            )
        return endpoint

    @property
    def _bedrock_available(self) -> bool:
        """Check if AWS Bedrock credentials are available.

        Checks for:
        1. Explicit AWS credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
        2. AWS session token (AWS_SESSION_TOKEN) - used with temporary credentials
        3. Custom Bedrock endpoint (ANTHROPIC_BEDROCK_BASE_URL) - Claude Code compatibility

        Note: boto3 also supports ~/.aws/credentials and instance profiles,
        but we can't easily check those without importing boto3.
        """
        # Check explicit credentials
        has_explicit_creds = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )

        # Check for custom Bedrock endpoint (Claude Code sets this)
        has_bedrock_endpoint = bool(os.environ.get("ANTHROPIC_BEDROCK_BASE_URL"))

        return has_explicit_creds or has_bedrock_endpoint

    def _get_bedrock_client(self) -> Any:
        """Get or create boto3 bedrock-runtime client (lazy initialization).

        Supports:
        - Standard boto3 credential chain (env vars, ~/.aws/credentials, instance profiles)
        - AWS_REGION / AWS_DEFAULT_REGION env vars (defaults to us-east-1)
        - ANTHROPIC_BEDROCK_BASE_URL for custom endpoint (Claude Code compatibility)

        Returns:
            boto3 bedrock-runtime client

        Raises:
            ImportError: If boto3 is not installed
        """
        if self._bedrock_client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError(
                    "boto3 is required for Bedrock support. Install with: pip install boto3"
                ) from e

            # Get AWS region from environment or use default
            region = os.environ.get("AWS_REGION") or os.environ.get(
                "AWS_DEFAULT_REGION", "us-east-1"
            )

            # Check for custom Bedrock endpoint (Claude Code sets this)
            bedrock_endpoint = os.environ.get("ANTHROPIC_BEDROCK_BASE_URL")

            if bedrock_endpoint:
                # Custom endpoint specified (Claude Code or custom setup)
                self._bedrock_client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=region,
                    endpoint_url=bedrock_endpoint,
                )
                logger.debug(
                    f"Initialized Bedrock client in region: {region} with custom endpoint: {bedrock_endpoint}"
                )
            else:
                # Standard Bedrock endpoint
                self._bedrock_client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=region,
                )
                logger.debug(f"Initialized Bedrock client in region: {region}")

        return self._bedrock_client

    async def generate_search_queries(
        self, natural_language_query: str, limit: int = 3
    ) -> list[str]:
        """Generate targeted search queries from natural language.

        Args:
            natural_language_query: User's natural language query
            limit: Maximum number of search queries to generate

        Returns:
            List of targeted search queries

        Raises:
            SearchError: If API call fails
        """
        system_prompt = """You are a code search expert. Your task is to convert natural language questions about code into targeted search queries.

Given a natural language query, generate {limit} specific search queries that will help find the relevant code.

Rules:
1. Each query should target a different aspect of the question
2. Use technical terms and identifiers when possible
3. Keep queries concise (3-7 words each)
4. Focus on code patterns, function names, class names, or concepts
5. Return ONLY the search queries, one per line, no explanations

Example:
Input: "where is the similarity_threshold parameter set?"
Output:
similarity_threshold default value
similarity_threshold configuration
SemanticSearchEngine init threshold"""

        user_prompt = f"""Natural language query: {natural_language_query}

Generate {limit} targeted search queries:"""

        try:
            messages = [
                {"role": "system", "content": system_prompt.format(limit=limit)},
                {"role": "user", "content": user_prompt},
            ]

            response = await self._chat_completion(messages)

            # Parse queries from response
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            queries = [q.strip() for q in content.strip().split("\n") if q.strip()]

            logger.debug(
                f"Generated {len(queries)} search queries from: '{natural_language_query}'"
            )

            return queries[:limit]

        except Exception as e:
            logger.error(f"Failed to generate search queries: {e}")
            raise SearchError(f"LLM query generation failed: {e}") from e

    async def analyze_and_rank_results(
        self,
        original_query: str,
        search_results: dict[str, list[Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Analyze search results and select the most relevant ones.

        Args:
            original_query: Original natural language query
            search_results: Dictionary mapping search queries to their results
            top_n: Number of top results to return

        Returns:
            List of ranked results with explanations

        Raises:
            SearchError: If API call fails
        """
        # Format results for LLM analysis
        results_summary = self._format_results_for_analysis(search_results)

        system_prompt = """You are a code search expert. Your task is to analyze search results and identify the most relevant ones for answering a user's question.

Given:
1. A natural language query
2. Multiple search results from different queries

Select the top {top_n} most relevant results that best answer the user's question.

For each selected result, provide:
1. Result identifier (e.g., "Query 1, Result 2")
2. Relevance level: "High", "Medium", or "Low"
3. Brief explanation (1-2 sentences) of why this result is relevant

Format your response as:
RESULT: [identifier]
RELEVANCE: [level]
EXPLANATION: [why this matches]

---

Only include the top {top_n} results."""

        user_prompt = f"""Original Question: {original_query}

Search Results:
{results_summary}

Select the top {top_n} most relevant results:"""

        try:
            messages = [
                {"role": "system", "content": system_prompt.format(top_n=top_n)},
                {"role": "user", "content": user_prompt},
            ]

            response = await self._chat_completion(messages)

            # Parse LLM response
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            ranked_results = self._parse_ranking_response(
                content, search_results, top_n
            )

            logger.debug(f"Ranked {len(ranked_results)} results from LLM analysis")

            return ranked_results

        except Exception as e:
            logger.error(f"Failed to analyze results: {e}")
            raise SearchError(f"LLM analysis failed: {e}") from e

    @classmethod
    async def is_ollama_available(cls) -> bool:
        """Check if Ollama is running and reachable.

        Returns:
            True if Ollama is available
        """
        from .ollama_detector import detect_ollama

        return await detect_ollama()

    def _build_request_headers(self) -> dict[str, str]:
        """Build HTTP headers for the current (non-Bedrock) provider.

        Returns:
            Dict of headers including Authorization and any provider-specific entries
        """
        if self.provider == "ollama":
            # Ollama's OpenAI-compat endpoint: no real auth required
            headers = {
                "Authorization": "Bearer ollama",
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

        # OpenRouter-specific headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/bobmatnyc/mcp-vector-search"
            headers["X-Title"] = "MCP Vector Search"

        return headers

    def _format_http_status_error(
        self, exc: httpx.HTTPStatusError, *, include_body_detail: bool = True
    ) -> str:
        """Build a user-friendly error message from an httpx HTTPStatusError.

        Args:
            exc: The exception to format
            include_body_detail: If True, attempt to extract error.message from JSON body

        Returns:
            Formatted error message string
        """
        provider_name = self.provider.capitalize()
        status_code = exc.response.status_code
        error_msg = f"{provider_name} API error (HTTP {status_code})"

        if include_body_detail:
            # Try to get more details from the response
            try:
                error_body = exc.response.json()
                error_detail = error_body.get("error", {}).get("message", "")
                if error_detail:
                    error_msg = f"{error_msg}: {error_detail}"
            except Exception:
                pass

        if status_code == 400:
            if self.provider == "ollama":
                error_msg = (
                    f"{error_msg}. Check model name with: ollama list\n"
                    f"Current model: {self.model}"
                )
            else:
                error_msg = f"{error_msg}. Check model name and request format."
        elif status_code == 401:
            env_var = (
                "OPENAI_API_KEY" if self.provider == "openai" else "OPENROUTER_API_KEY"
            )
            error_msg = (
                f"Invalid {provider_name} API key. "
                f"Please check {env_var} environment variable."
            )
        elif status_code == 429:
            error_msg = (
                f"{provider_name} API rate limit exceeded. Please wait and try again."
            )
        elif status_code >= 500:
            error_msg = f"{provider_name} API server error. Please try again later."

        return error_msg

    async def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Make chat completion request to OpenAI, OpenRouter, Bedrock, or Ollama API.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            API response dictionary

        Raises:
            SearchError: If API request fails
        """
        # Route to Bedrock if that's the provider
        if self.provider == "bedrock":
            return await self._bedrock_chat_completion(messages)

        headers = self._build_request_headers()
        payload = {
            "model": self.model,
            "messages": messages,
        }
        provider_name = self.provider.capitalize()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self._get_endpoint(),
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()
                return response.json()

        except httpx.ConnectError as e:
            if self.provider == "ollama":
                raise SearchError(
                    "Ollama not running. Start it with: ollama serve\n"
                    "Then pull a model: ollama pull gemma3"
                ) from e
            logger.error(f"{provider_name} connection failed: {e}")
            raise SearchError(f"Cannot connect to {provider_name}: {e}") from e

        except httpx.TimeoutException as e:
            logger.error(f"{provider_name} API timeout after {self.timeout}s")
            raise SearchError(
                f"LLM request timed out after {self.timeout} seconds. "
                "Try a simpler query or check your network connection."
            ) from e

        except httpx.HTTPStatusError as e:
            error_msg = self._format_http_status_error(e)
            logger.error(error_msg)
            raise SearchError(error_msg) from e

        except Exception as e:
            logger.error(f"{provider_name} API request failed: {e}")
            raise SearchError(f"LLM request failed: {e}") from e

    def _build_bedrock_request_params(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Build the Bedrock Converse API request_params dict from OpenAI-format messages.

        Splits out user/assistant turns into Bedrock content blocks and lifts the
        first system message into the dedicated ``system`` field.
        """
        # Convert messages to Bedrock format
        bedrock_messages = []
        for msg in messages:
            role = msg["role"]
            # Bedrock uses "user" and "assistant" (no "system" in messages)
            if role in ("user", "assistant"):
                bedrock_messages.append(
                    {"role": role, "content": [{"text": msg["content"]}]}
                )

        # Extract system message if present
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        system_content = None
        if system_messages:
            system_content = [{"text": system_messages[0]["content"]}]

        # Build Bedrock request
        request_params: dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": 4096,
                "temperature": 0.7,
            },
        }

        # Add system message if present
        if system_content:
            request_params["system"] = system_content

        return request_params

    def _format_bedrock_error(self, exc: Exception) -> str:
        """Translate a Bedrock SDK exception into a user-friendly error string."""
        error_msg = str(exc)

        # Parse common Bedrock errors
        if "AccessDeniedException" in error_msg:
            return (
                "AWS credentials invalid or insufficient permissions for Bedrock. "
                "Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )
        if "ValidationException" in error_msg:
            return f"Invalid Bedrock request: {error_msg}"
        if "ThrottlingException" in error_msg:
            return "Bedrock API rate limit exceeded. Please wait and try again."
        if "ModelNotReadyException" in error_msg:
            return f"Bedrock model {self.model} is not ready or not available in your region."
        if "ResourceNotFoundException" in error_msg:
            return f"Bedrock model {self.model} not found. Check model ID and region."
        return error_msg

    async def _bedrock_chat_completion(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Make chat completion request to AWS Bedrock using Converse API.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            API response dictionary in OpenAI format (for compatibility)

        Raises:
            SearchError: If Bedrock API request fails
        """
        try:
            request_params = self._build_bedrock_request_params(messages)

            # Run boto3 call in executor (boto3 is synchronous)
            loop = asyncio.get_event_loop()
            bedrock_client = self._get_bedrock_client()

            response = await loop.run_in_executor(
                None,
                lambda: bedrock_client.converse(**request_params),
            )

            # Convert Bedrock response to OpenAI format for compatibility
            content = response["output"]["message"]["content"][0]["text"]

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": response.get("stopReason", "stop"),
                    }
                ],
                "usage": response.get("usage", {}),
            }

        except Exception as e:
            logger.error(f"Bedrock API request failed: {e}")
            error_msg = self._format_bedrock_error(e)
            raise SearchError(f"Bedrock request failed: {error_msg}") from e

    def _format_results_for_analysis(self, search_results: dict[str, list[Any]]) -> str:
        """Format search results for LLM analysis.

        Args:
            search_results: Dictionary mapping search queries to their results

        Returns:
            Formatted string representation of results
        """
        formatted = []

        for i, (query, results) in enumerate(search_results.items(), 1):
            formatted.append(f"\n=== Query {i}: {query} ===")

            if not results:
                formatted.append("  No results found.")
                continue

            for j, result in enumerate(results[:5], 1):  # Top 5 per query
                # Extract key information from SearchResult
                file_path = str(result.file_path)
                similarity = result.similarity_score
                content_preview = result.content[:150].replace("\n", " ")

                formatted.append(
                    f"\n  Result {j}:\n"
                    f"    File: {file_path}\n"
                    f"    Similarity: {similarity:.3f}\n"
                    f"    Preview: {content_preview}..."
                )

                if result.function_name:
                    formatted.append(f"    Function: {result.function_name}")
                if result.class_name:
                    formatted.append(f"    Class: {result.class_name}")

        return "\n".join(formatted)

    @staticmethod
    def _parse_ranking_blocks(llm_response: str) -> list[dict[str, str]]:
        """Parse RESULT:/RELEVANCE:/EXPLANATION: line-blocks from LLM output."""
        ranked: list[dict[str, str]] = []
        current_result: dict[str, str] = {}

        for line in llm_response.split("\n"):
            line = line.strip()

            if line.startswith("RESULT:"):
                if current_result:
                    ranked.append(current_result)
                current_result = {"identifier": line.replace("RESULT:", "").strip()}

            elif line.startswith("RELEVANCE:"):
                current_result["relevance"] = line.replace("RELEVANCE:", "").strip()

            elif line.startswith("EXPLANATION:"):
                current_result["explanation"] = line.replace("EXPLANATION:", "").strip()

        # Add last result
        if current_result:
            ranked.append(current_result)

        return ranked

    @staticmethod
    def _resolve_ranking_identifier(
        identifier: str, search_results: dict[str, list[Any]]
    ) -> tuple[str, Any] | None:
        """Resolve a "Query N, Result M" identifier to (query, result) tuple.

        Returns None if the identifier cannot be parsed or indices are out of range.
        """
        try:
            parts = identifier.split(",")
            query_part = parts[0].replace("Query", "").strip()
            result_part = parts[1].replace("Result", "").strip()

            # Handle case where LLM includes filename in parentheses: "5 (config.py)"
            # Extract just the number
            query_match = re.match(r"(\d+)", query_part)
            result_match = re.match(r"(\d+)", result_part)

            if not query_match or not result_match:
                logger.warning(
                    f"Could not extract numbers from identifier '{identifier}'"
                )
                return None

            query_idx = int(query_match.group(1)) - 1
            result_idx = int(result_match.group(1)) - 1

            queries = list(search_results.keys())
            if query_idx >= len(queries):
                return None

            query = queries[query_idx]
            results = search_results[query]
            if result_idx >= len(results):
                return None

            return query, results[result_idx]

        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse result identifier '{identifier}': {e}")
            return None

    def _parse_ranking_response(
        self,
        llm_response: str,
        search_results: dict[str, list[Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Parse LLM ranking response into structured results.

        Args:
            llm_response: Raw LLM response text
            search_results: Original search results dictionary
            top_n: Maximum number of results to return

        Returns:
            List of ranked results with metadata
        """
        ranked = self._parse_ranking_blocks(llm_response)

        # Map identifiers back to actual SearchResult objects
        enriched_results: list[dict[str, Any]] = []

        for item in ranked[:top_n]:
            identifier = item.get("identifier", "")
            resolved = self._resolve_ranking_identifier(identifier, search_results)
            if resolved is None:
                continue
            query, actual_result = resolved
            enriched_results.append(
                {
                    "result": actual_result,
                    "query": query,
                    "relevance": item.get("relevance", "Medium"),
                    "explanation": item.get("explanation", "Relevant to query"),
                }
            )

        return enriched_results

    async def detect_intent(self, query: str) -> IntentType:
        """Detect user intent from query.

        Args:
            query: User's natural language query

        Returns:
            Intent type: "find", "answer", or "analyze"

        Raises:
            SearchError: If API call fails
        """
        system_prompt = """You are a code search intent classifier. Classify the user's query into ONE of these categories:

1. "find" - User wants to locate/search for something in the codebase
   Examples: "where is X", "find the function that", "show me the code for", "locate X"

2. "answer" - User wants an explanation/answer about the codebase
   Examples: "what does this do", "how does X work", "explain the architecture", "why is X used"

3. "analyze" - User wants analysis of code quality, metrics, complexity, or smells
   Examples: "what's complex", "code smells", "cognitive complexity", "quality issues",
   "dependencies", "coupling", "circular dependencies", "getting worse", "improving",
   "analyze the complexity", "find the worst code", "most complex functions"

Return ONLY the word "find", "answer", or "analyze" with no other text."""

        user_prompt = f"""Query: {query}

Intent:"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self._chat_completion(messages)

            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            intent = content.strip().lower()

            if intent not in ("find", "answer", "analyze"):
                # Default to find if unclear
                logger.warning(
                    f"Unclear intent '{intent}' for query '{query}', defaulting to 'find'"
                )
                return "find"

            logger.debug(f"Detected intent '{intent}' for query: '{query}'")
            return intent  # type: ignore

        except Exception as e:
            logger.error(f"Failed to detect intent: {e}, defaulting to 'find'")
            return "find"

    @staticmethod
    def _parse_sse_line(line: str) -> str | None:
        """Parse a single SSE line into emittable content.

        Args:
            line: Raw line from the SSE stream

        Returns:
            - None if the line should be skipped (empty, comment, malformed JSON, no content)
            - "__DONE__" sentinel if the stream signalled end-of-stream
            - The content text chunk otherwise (may be empty string if delta lacked content)
        """
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith(":"):
            return None

        # Parse SSE format: "data: {json}"
        if not line.startswith("data: "):
            return None

        data = line[6:]  # Remove "data: " prefix

        # Check for end of stream
        if data == "[DONE]":
            return "__DONE__"

        try:
            chunk = json.loads(data)
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                return content
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse SSE chunk: {e}")
            return None

    async def stream_chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream chat completion response chunk by chunk.

        Args:
            messages: List of message dictionaries with role and content

        Yields:
            Text chunks from the streaming response

        Raises:
            SearchError: If API request fails
        """
        # Route to Bedrock streaming if that's the provider
        if self.provider == "bedrock":
            async for chunk in self._bedrock_stream_chat_completion(messages):
                yield chunk
            return

        headers = self._build_request_headers()

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        provider_name = self.provider.capitalize()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", self._get_endpoint(), headers=headers, json=payload
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        parsed = self._parse_sse_line(line)
                        if parsed is None:
                            continue
                        if parsed == "__DONE__":
                            break
                        if parsed:
                            yield parsed

        except httpx.TimeoutException as e:
            logger.error(f"{provider_name} API timeout after {self.timeout}s")
            raise SearchError(
                f"LLM request timed out after {self.timeout} seconds. "
                "Try a simpler query or check your network connection."
            ) from e

        except httpx.HTTPStatusError as e:
            error_msg = self._format_stream_status_error(e)
            logger.error(error_msg)
            raise SearchError(error_msg) from e

        except Exception as e:
            logger.error(f"{provider_name} streaming request failed: {e}")
            raise SearchError(f"LLM streaming failed: {e}") from e

    def _format_stream_status_error(self, exc: httpx.HTTPStatusError) -> str:
        """Format an HTTPStatusError raised during streaming.

        Mirrors the legacy behaviour: handles 401/429/500+, no body-detail
        extraction and no 400-specific message.
        """
        provider_name = self.provider.capitalize()
        status_code = exc.response.status_code
        error_msg = f"{provider_name} API error (HTTP {status_code})"

        if status_code == 401:
            env_var = (
                "OPENAI_API_KEY" if self.provider == "openai" else "OPENROUTER_API_KEY"
            )
            error_msg = (
                f"Invalid {provider_name} API key. "
                f"Please check {env_var} environment variable."
            )
        elif status_code == 429:
            error_msg = (
                f"{provider_name} API rate limit exceeded. Please wait and try again."
            )
        elif status_code >= 500:
            error_msg = f"{provider_name} API server error. Please try again later."

        return error_msg

    async def _bedrock_stream_chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream chat completion from AWS Bedrock using Converse Stream API.

        Args:
            messages: List of message dictionaries with role and content

        Yields:
            Text chunks from the streaming response

        Raises:
            SearchError: If Bedrock streaming request fails
        """
        try:
            request_params = self._build_bedrock_request_params(messages)

            # Run streaming call in executor
            loop = asyncio.get_event_loop()
            bedrock_client = self._get_bedrock_client()

            response = await loop.run_in_executor(
                None,
                lambda: bedrock_client.converse_stream(**request_params),
            )

            # Process streaming response
            stream = response.get("stream")
            if stream:
                for event in stream:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]["delta"]
                        if "text" in delta:
                            yield delta["text"]

        except Exception as e:
            logger.error(f"Bedrock streaming request failed: {e}")
            error_msg = str(e)

            if "AccessDeniedException" in error_msg:
                error_msg = (
                    "AWS credentials invalid or insufficient permissions for Bedrock."
                )
            elif "ThrottlingException" in error_msg:
                error_msg = (
                    "Bedrock API rate limit exceeded. Please wait and try again."
                )

            raise SearchError(f"Bedrock streaming failed: {error_msg}") from e

    async def generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate answer to user question using codebase context.

        Args:
            query: User's question
            context: Relevant code context from search results
            conversation_history: Previous conversation messages (optional)

        Returns:
            LLM response text

        Raises:
            SearchError: If API call fails
        """
        system_prompt = f"""You are a helpful code assistant analyzing a codebase. Answer the user's questions based on the provided code context.

Code Context:
{context}

Guidelines:
- Be concise but thorough in explanations
- Reference specific functions, classes, or files when relevant
- Use code examples from the context when helpful
- If the context doesn't contain enough information, say so
- Use markdown formatting for code snippets"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current query
        messages.append({"role": "user", "content": query})

        try:
            response = await self._chat_completion(messages)
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            logger.debug(f"Generated answer for query: '{query}'")
            return content

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise SearchError(f"Failed to generate answer: {e}") from e

    # ------------------------------------------------------------------
    # Ollama prompt-engineering tool-call helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tools_to_system_prompt(tools: list[dict[str, Any]]) -> str:
        """Convert OpenAI tool-schema list into a system-prompt block for Ollama.

        Gemma and similar models don't support native function calling.
        We describe the tools in the system prompt and ask the model to
        output structured XML tags that we can parse.

        Format injected into system prompt:
            You have access to tools. To call a tool output EXACTLY:
            <tool_call>{"name": "...", "arguments": {...}}</tool_call>

            After each tool result is shown you may call another tool or give
            your final answer (no tag).

        Args:
            tools: OpenAI-format tool definitions list

        Returns:
            Formatted string block to append to the system prompt
        """
        if not tools:
            return ""

        lines = [
            "",
            "## Available Tools",
            "",
            "You have access to the following tools. To use a tool output EXACTLY this format "
            "(one per response, nothing before or after the tag on the same line):",
            "",
            '    <tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>',
            "",
            "After I show you the tool result, you can call another tool or give your final answer.",
            "When you are done gathering information, answer the user's question directly without a tag.",
            "",
            "### Tool Definitions",
            "",
        ]

        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            lines.append(f"**{name}**: {desc}")
            if props:
                param_parts = []
                for pname, pdef in props.items():
                    ptype = pdef.get("type", "any")
                    pdesc = pdef.get("description", "")
                    req_marker = " (required)" if pname in required else ""
                    param_parts.append(f"  - {pname}: {ptype}{req_marker} — {pdesc}")
                lines.extend(param_parts)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _inject_tools_into_messages(
        messages: list[dict[str, Any]], tool_prompt: str
    ) -> list[dict[str, Any]]:
        """Append the tool-prompt block to the last system message (or prepend one).

        Args:
            messages: Conversation message list
            tool_prompt: Tool description block

        Returns:
            Modified message list (new list, originals not mutated)
        """
        if not tool_prompt:
            return messages

        result = list(messages)

        # Find the last system message
        last_sys_idx = None
        for i, msg in enumerate(result):
            if msg.get("role") == "system":
                last_sys_idx = i

        if last_sys_idx is not None:
            existing = result[last_sys_idx]
            result[last_sys_idx] = {
                **existing,
                "content": existing["content"] + tool_prompt,
            }
        else:
            result.insert(0, {"role": "system", "content": tool_prompt.strip()})

        return result

    @staticmethod
    def _make_tool_call(call_id: str, name: str, args: Any) -> dict[str, Any]:
        """Construct an OpenAI-format tool_call dict from a name/args pair."""
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        }

    @staticmethod
    def _parse_tool_calls_xml(content: str) -> list[dict[str, Any]]:
        """Parse ``<tool_call>{...}</tool_call>`` XML-tagged tool calls."""
        calls: list[dict[str, Any]] = []
        xml_pattern = re.compile(
            rf"{re.escape(_TOOL_CALL_OPEN)}\s*(.*?)\s*{re.escape(_TOOL_CALL_CLOSE)}",
            re.DOTALL,
        )
        for match_idx, m in enumerate(xml_pattern.finditer(content)):
            raw = m.group(1).strip()
            try:
                parsed = json.loads(raw)
                name = parsed.get("name", "")
                args = parsed.get("arguments", parsed.get("args", {}))
                if name:
                    calls.append(
                        LLMClient._make_tool_call(
                            f"ollama_call_{match_idx}", name, args
                        )
                    )
            except json.JSONDecodeError as exc:
                logger.debug(f"Could not parse tool_call XML block: {exc}\nRaw: {raw}")
        return calls

    @staticmethod
    def _collect_json_candidates(content: str, stripped: str) -> list[str]:
        """Collect JSON candidate substrings from raw model output.

        Handles markdown code-fences and embedded JSON objects in prose.
        """
        code_fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", stripped)
        if code_fence:
            return [code_fence.group(1).strip()]

        candidates: list[str] = [stripped]
        # Also scan for embedded JSON objects (e.g. text\n{...} )
        for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", content, re.DOTALL):
            candidate = m.group(0).strip()
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _parse_tool_calls_bare_json(
        content: str, stripped: str
    ) -> list[dict[str, Any]]:
        """Parse a bare JSON object (or fenced block) with name/arguments keys."""
        for json_candidate in LLMClient._collect_json_candidates(content, stripped):
            try:
                parsed = json.loads(json_candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            # Must look like a tool call: has "name" (str) and "arguments" (dict)
            if (
                isinstance(parsed, dict)
                and isinstance(parsed.get("name"), str)
                and parsed["name"]
                and isinstance(parsed.get("arguments", parsed.get("args")), dict)
            ):
                name = parsed["name"]
                args = parsed.get("arguments", parsed.get("args", {}))
                return [LLMClient._make_tool_call("ollama_call_0", name, args)]
        return []

    @staticmethod
    def _parse_tool_calls_fn_json(stripped: str) -> list[dict[str, Any]]:
        """Parse ``function_name {json_args}`` single-line format (qwen2.5-coder)."""
        fn_json_pattern = re.compile(
            r"^([a-zA-Z_][a-zA-Z0-9_]*)\s+(\{.*\})\s*$",
            re.DOTALL,
        )
        fn_match = fn_json_pattern.match(stripped)
        if not fn_match:
            return []

        fn_name = fn_match.group(1)
        raw_args = fn_match.group(2).strip()
        try:
            args = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(args, dict):
            return []
        return [LLMClient._make_tool_call("ollama_call_0", fn_name, args)]

    @staticmethod
    def _parse_ollama_tool_calls(content: str) -> list[dict[str, Any]]:
        """Extract tool call dicts from model output.

        Handles three output formats used by different local models:
        1. ``<tool_call>{"name": ..., "arguments": ...}</tool_call>`` XML tags
           (prompt-engineered models)
        2. Bare JSON object with ``"name"`` and ``"arguments"`` keys
           (qwen2.5-coder and similar models that ignore the tools parameter
           but still output structured JSON)
        3. JSON code-fence blocks containing the above structure

        Args:
            content: Raw text response from the model

        Returns:
            List of parsed tool call dicts (OpenAI tool_calls format), may be empty
        """
        # --- Format 1: <tool_call>...</tool_call> XML tags ---
        calls = LLMClient._parse_tool_calls_xml(content)
        if calls:
            return calls

        stripped = content.strip()

        # --- Format 2: bare JSON object or JSON code-fence block ---
        calls = LLMClient._parse_tool_calls_bare_json(content, stripped)
        if calls:
            return calls

        # --- Format 3: "function_name {json_args}" on a single line ---
        return LLMClient._parse_tool_calls_fn_json(stripped)

    async def _ollama_native_chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Native function-calling for Ollama via the OpenAI-compatible endpoint.

        Ollama's /v1/chat/completions endpoint accepts ``tools`` in the request
        body (same schema as OpenAI) for models that support it (mistral,
        qwen2.5-coder, llama3, gemma3, etc.).

        Args:
            messages: Conversation messages
            tools: OpenAI-format tool list

        Returns:
            OpenAI-compatible response dict

        Raises:
            SearchError: If request fails
        """
        headers = {
            "Authorization": "Bearer ollama",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self._get_endpoint(), headers=headers, json=payload
            )
            response.raise_for_status()
            return response.json()

    async def _try_ollama_native_call(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Attempt the Ollama native tool-calling endpoint, returning None on failure.

        Returns:
            Response dict on success, or None if the call failed in a way that
            indicates we should fall back to prompt-engineering.
        """
        try:
            return await self._ollama_native_chat_with_tools(messages, tools)
        except httpx.HTTPStatusError as exc:
            # Some older Ollama builds / models reject the tools parameter
            if exc.response.status_code in (400, 422):
                logger.debug(
                    f"Ollama native tool calling failed ({exc.response.status_code}), "
                    "falling back to prompt-engineering approach"
                )
                return None
            raise
        except Exception:
            return None

    def _handle_ollama_native_response(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a successful native-call response, extracting tool_calls if any.

        Falls back to scanning content for ``<tool_call>`` XML tags when the
        native response did not include native tool_calls.
        """
        msg = response.get("choices", [{}])[0].get("message", {})
        if msg.get("tool_calls"):
            logger.debug("Ollama returned native tool_calls — using them directly")
            return response

        # Native call succeeded but no tool_calls — check for XML fallback tags
        content = msg.get("content", "") or ""
        xml_calls = self._parse_ollama_tool_calls(content)
        if xml_calls:
            logger.debug(
                "Ollama response contained <tool_call> XML tags — parsed as fallback"
            )
            clean_content = re.sub(
                rf"{re.escape(_TOOL_CALL_OPEN)}.*?{re.escape(_TOOL_CALL_CLOSE)}",
                "",
                content,
                flags=re.DOTALL,
            ).strip()
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": clean_content or None,
                            "tool_calls": xml_calls,
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }

        # No tool calls at all — plain text answer
        return response

    @staticmethod
    def _normalize_messages_for_prompt_fallback(
        patched_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Flatten tool/tool_call messages into plain assistant/user text.

        Ollama's prompt-engineering fallback can't handle the OpenAI tool/role
        ``tool`` messages or ``assistant.tool_calls`` arrays, so rewrite them
        as <tool_call> tagged assistant messages and ``[Tool result]`` user
        messages.
        """
        normalized: list[dict[str, Any]] = []
        for msg in patched_messages:
            role = msg.get("role", "")
            if role == "tool":
                normalized.append(
                    {
                        "role": "user",
                        "content": f"[Tool result]\n{msg.get('content', '')}",
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                tc = msg["tool_calls"][0]["function"]
                normalized.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"{_TOOL_CALL_OPEN}"
                            f'{{"name": "{tc["name"]}", '
                            f'"arguments": {tc["arguments"]}}}'
                            f"{_TOOL_CALL_CLOSE}"
                        ),
                    }
                )
            else:
                normalized.append(msg)
        return normalized

    async def _ollama_chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Tool-calling for Ollama: native API first, XML fallback second.

        Strategy:
        1. Try Ollama's native ``tools:`` parameter (OpenAI-compatible format).
           Models like qwen2.5-coder, llama3, gemma3 honour this natively and
           return ``tool_calls`` in the response.
        2. If the response has no ``tool_calls`` AND the content text contains
           ``<tool_call>`` tags, parse those tags as a fallback (handles older
           models that were prompted with XML).
        3. If the native call fails with a 400/422 (model doesn't support
           tools natively), fall back to the prompt-engineering approach.

        Args:
            messages: Conversation messages
            tools: OpenAI-format tool list

        Returns:
            OpenAI-compatible response dict (with tool_calls if a tool was requested)
        """
        # --- Attempt 1: native function calling ---
        response = await self._try_ollama_native_call(messages, tools)

        if response is not None:
            return self._handle_ollama_native_response(response)

        # --- Attempt 2: prompt-engineering fallback ---
        logger.debug("Using prompt-engineering tool-call fallback for Ollama")
        tool_prompt = self._tools_to_system_prompt(tools)
        patched_messages = self._inject_tools_into_messages(messages, tool_prompt)

        # Strip any existing tool / tool_result messages that Ollama can't handle;
        # flatten them into assistant/user text instead.
        normalized = self._normalize_messages_for_prompt_fallback(patched_messages)

        fallback_response = await self._chat_completion(normalized)

        content = (
            fallback_response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        )
        tool_calls = self._parse_ollama_tool_calls(content)

        if tool_calls:
            clean_content = re.sub(
                rf"{re.escape(_TOOL_CALL_OPEN)}.*?{re.escape(_TOOL_CALL_CLOSE)}",
                "",
                content,
                flags=re.DOTALL,
            ).strip()
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": clean_content or None,
                            "tool_calls": tool_calls,
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }

        return fallback_response

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Chat completion with tool/function calling support.

        Args:
            messages: List of message dictionaries
            tools: List of tool definitions

        Returns:
            API response with tool calls or final message

        Raises:
            SearchError: If API request fails

        Note:
            Bedrock tool calling is not yet implemented. Falls back to regular chat.
            Ollama tries native function calling first, then XML prompt-engineering.
        """
        # Ollama: native function calling with XML fallback
        if self.provider == "ollama":
            return await self._ollama_chat_with_tools(messages, tools)

        # TODO: Implement Bedrock tool calling when needed
        # Bedrock uses a different tool format (toolConfig) than OpenAI
        if self.provider == "bedrock":
            logger.warning(
                "Tool calling not yet implemented for Bedrock, falling back to regular chat"
            )
            return await self._chat_completion(messages)

        headers = self._build_request_headers()

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }

        provider_name = self.provider.capitalize()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self._get_endpoint(),
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException as e:
            logger.error(f"{provider_name} API timeout after {self.timeout}s")
            raise SearchError(
                f"LLM request timed out after {self.timeout} seconds."
            ) from e

        except httpx.HTTPStatusError as e:
            error_msg = self._format_chat_with_tools_status_error(e)
            logger.error(error_msg)
            raise SearchError(error_msg) from e

        except Exception as e:
            logger.error(f"{provider_name} API request failed: {e}")
            raise SearchError(f"LLM request failed: {e}") from e

    def _format_chat_with_tools_status_error(self, exc: httpx.HTTPStatusError) -> str:
        """Format an HTTPStatusError using chat_with_tools' (terser) message style.

        Distinct from :meth:`_format_http_status_error` to preserve the exact
        legacy phrasing of the chat_with_tools error path.
        """
        provider_name = self.provider.capitalize()
        status_code = exc.response.status_code
        error_msg = f"{provider_name} API error (HTTP {status_code})"

        # Try to get more details from the response
        try:
            error_body = exc.response.json()
            error_detail = error_body.get("error", {}).get("message", "")
            if error_detail:
                error_msg = f"{error_msg}: {error_detail}"
        except Exception:
            pass

        if status_code == 400:
            error_msg = f"{error_msg}. Check model name and request format."
        elif status_code == 401:
            env_var = (
                "OPENAI_API_KEY" if self.provider == "openai" else "OPENROUTER_API_KEY"
            )
            error_msg = f"Invalid {provider_name} API key. Check {env_var}."
        elif status_code == 429:
            error_msg = f"{provider_name} API rate limit exceeded."
        elif status_code >= 500:
            error_msg = f"{provider_name} API server error."

        return error_msg
