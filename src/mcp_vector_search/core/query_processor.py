"""Query preprocessing and optimization for semantic search."""

import re

_IDENTIFIER_PATTERNS = [
    re.compile(r"\b[\w-]+\.[\w-]+"),  # dotted: getstream.io, io.sentry
    # CamelCase / camelCase: an internal lower→upper boundary OR a leading
    # capital followed by a lowercase chunk and another uppercase chunk.
    # Matches both "StreamApp" and "getStream" but not plain "Login".
    re.compile(r"\b(?:[a-z][\w]*[A-Z][\w]*|[A-Z][a-z]+[A-Z][\w]*)\b"),
    re.compile(r"@[\w][\w/-]+"),  # npm scoped: @tanstack/query (no \b before @)
    re.compile(
        r"\b[\w][\w]*-[\w][\w]*-[\w][\w]*\b"
    ),  # multi-hyphen: react-activity-feed
]
_PACKAGE_KEYWORDS = frozenset(
    ["sdk", "npm", "pip", "pypi", "crate", "package", "library", "lib"]
)


def is_identifier_query(query: str) -> bool:
    """Return True if the query looks like an SDK name, package, or code identifier.

    These queries are better served by BM25 than vector similarity, so callers
    should lower hybrid_alpha (e.g. 0.2) when this returns True.
    """
    for pattern in _IDENTIFIER_PATTERNS:
        if pattern.search(query):
            return True
    return any(w in _PACKAGE_KEYWORDS for w in query.lower().split())


# Patterns that signal a natural-language (NL) query as opposed to a code query.
# NL queries like "functions that handle authentication" match docstrings better.
_NL_QUESTION_WORDS = frozenset(
    [
        "what",
        "where",
        "how",
        "why",
        "when",
        "which",
        "who",
        "find",
        "show",
        "get",
        "list",
        "describe",
        "explain",
        "return",
        "returns",
        "purpose",
        "does",
        "do",
        "handle",
        "handles",
        "implement",
        "implements",
        "used",
        "use",
        "fetch",
        "check",
        "validate",
        "process",
        "manage",
        "create",
        "build",
        "compute",
        "calculate",
    ]
)

_CODE_OPERATOR_PATTERN = re.compile(
    r"[(){}\[\]=<>!&|+\-*/\\;:@#%^~`]"  # code operators/punctuation
)
_SNAKE_CASE_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b")


def is_nl_query(query: str) -> bool:
    """Return True if the query looks like a natural-language question/description.

    NL queries match docstrings and comments better than raw code bodies.
    Code queries (with operators, snake_case, CamelCase identifiers) match code better.

    Examples:
        is_nl_query("functions that handle authentication") → True
        is_nl_query("retry_count += 1") → False
        is_nl_query("find retry with exponential backoff") → True
        is_nl_query("AsyncIterator[StreamApp]") → False
    """
    # Queries with code operators are code queries
    if _CODE_OPERATOR_PATTERN.search(query):
        return False

    # Queries with CamelCase or snake_case identifiers are code queries
    if _IDENTIFIER_PATTERNS[1].search(query):  # CamelCase pattern
        return False
    if _SNAKE_CASE_PATTERN.search(query):
        return False

    # Queries that start with or contain NL question/intent words are NL queries
    words = query.lower().split()
    if not words:
        return False
    if words[0] in _NL_QUESTION_WORDS:
        return True
    if any(w in _NL_QUESTION_WORDS for w in words):
        return True

    # Queries with 4+ words and no code patterns are likely NL
    return len(words) >= 4


class QueryProcessor:
    """Handles query preprocessing, expansion, and adaptive threshold calculation."""

    # Query expansion constants (class-level for performance)
    _QUERY_EXPANSIONS = {
        # Common abbreviations
        "auth": "authentication authorize login",
        "db": "database data storage",
        "api": "application programming interface endpoint",
        "ui": "user interface frontend view",
        "util": "utility helper function",
        "config": "configuration settings options",
        "async": "asynchronous await promise",
        "sync": "synchronous blocking",
        "func": "function method",
        "var": "variable",
        "param": "parameter argument",
        "init": "initialize setup create",
        "parse": "parsing parser analyze",
        "validate": "validation check verify",
        "handle": "handler process manage",
        "error": "exception failure bug",
        "test": "testing unittest spec",
        "mock": "mocking stub fake",
        "log": "logging logger debug",
        # Programming concepts
        "class": "class object type",
        "method": "method function procedure",
        "property": "property attribute field",
        "import": "import require include",
        "export": "export module public",
        "return": "return yield output",
        "loop": "loop iterate for while",
        "condition": "condition if else branch",
        "array": "array list collection",
        "string": "string text character",
        "number": "number integer float",
        "boolean": "boolean true false",
    }

    def __init__(self, base_threshold: float = 0.2) -> None:
        """Initialize query processor.

        Args:
            base_threshold: Default similarity threshold
        """
        self.base_threshold = base_threshold

    def preprocess_query(self, query: str) -> str:
        """Preprocess search query for better results.

        Args:
            query: Raw search query

        Returns:
            Processed query with expansions
        """
        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query.strip())

        # Use class-level query expansions (no dict creation overhead)
        words = query.lower().split()
        expanded_words = []

        for word in words:
            # Add original word
            expanded_words.append(word)

            # Add expansions if available
            if word in self._QUERY_EXPANSIONS:
                expanded_words.extend(self._QUERY_EXPANSIONS[word].split())

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in expanded_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)

        return " ".join(unique_words)

    def get_adaptive_threshold(self, query: str) -> float:
        """Get adaptive similarity threshold based on query characteristics.

        Args:
            query: Search query

        Returns:
            Adaptive similarity threshold
        """
        base_threshold = self.base_threshold
        query_lower = query.lower()
        words = query.split()

        # Adjust threshold based on query characteristics

        # 1. Single word queries - lower threshold for broader results
        if len(words) == 1:
            return max(0.01, base_threshold - 0.29)

        # 2. Very specific technical terms - lower threshold
        technical_terms = [
            "javascript",
            "typescript",
            "python",
            "java",
            "cpp",
            "rust",
            "go",
            "function",
            "class",
            "method",
            "variable",
            "import",
            "export",
            "async",
            "await",
            "promise",
            "callback",
            "api",
            "database",
            "parser",
            "compiler",
            "interpreter",
            "syntax",
            "semantic",
            "mcp",
            "gateway",
            "server",
            "client",
            "protocol",
        ]

        if any(term in query_lower for term in technical_terms):
            return max(0.01, base_threshold - 0.29)

        # 3. Short queries (2-3 words) - slightly lower threshold
        if len(words) <= 3:
            return max(0.1, base_threshold - 0.1)

        # 4. Long queries (>6 words) - higher threshold for precision
        if len(words) > 6:
            return min(0.8, base_threshold + 0.1)

        # 5. Queries with exact identifiers (CamelCase, snake_case)
        if re.search(r"\b[A-Z][a-zA-Z]*\b", query) or "_" in query:
            return max(0.05, base_threshold - 0.25)

        # 6. Common programming patterns
        if any(pattern in query for pattern in ["()", ".", "->", "=>", "::"]):
            return max(0.25, base_threshold - 0.1)

        return max(0.15, base_threshold)
