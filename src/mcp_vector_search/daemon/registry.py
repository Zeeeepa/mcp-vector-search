"""LRU registry of opened SemanticSearchEngine instances per project path."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from loguru import logger

from .protocol import ModelMismatchError


class IndexRegistry:
    """Maintain an LRU cache of open search engines, one per project path.

    Each entry is a (SemanticSearchEngine, asyncio.Lock) tuple. The lock is
    held during searches and during index mutations so the daemon never serves
    a query against an index mid-write.
    """

    def __init__(self, max_indexes: int = 5, loaded_model: str | None = None) -> None:
        self.max_indexes = max_indexes
        self.loaded_model = loaded_model
        # OrderedDict gives us O(1) LRU semantics via move_to_end / popitem(last=False)
        self._cache: OrderedDict[str, tuple[Any, asyncio.Lock]] = OrderedDict()
        # Cache the bundle so we can close the database properly during eviction
        self._bundles: dict[str, Any] = {}
        # Lock that protects mutation of _cache itself
        self._registry_lock = asyncio.Lock()

    @staticmethod
    def _canonical(project_path: str) -> str:
        return str(Path(project_path).expanduser().resolve())

    def _check_model(self, project_path: Path) -> None:
        """Compare stored embedding_model in metadata against daemon's loaded model.

        Raises ModelMismatchError if they differ. If metadata is missing or has
        no embedding_model recorded, we trust the daemon's model.
        """
        if not self.loaded_model:
            return
        meta_file = project_path / ".mcp-vector-search" / "index_metadata.json"
        if not meta_file.exists():
            return
        try:
            with open(meta_file) as f:
                raw = json.load(f)
        except Exception as e:  # pragma: no cover - corrupt metadata
            logger.warning(f"Could not read {meta_file}: {e}")
            return
        stored = raw.get("embedding_model")
        if stored and stored != self.loaded_model:
            raise ModelMismatchError(stored, self.loaded_model)

    async def get_or_open(self, project_path: str) -> tuple[Any, asyncio.Lock]:
        """Return (engine, lock) for the given project, opening if needed.

        Performs LRU eviction when the cache is over capacity. Raises
        ModelMismatchError if the project's stored embedding_model doesn't
        match the daemon's loaded model.
        """
        key = self._canonical(project_path)

        async with self._registry_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            project_root = Path(key)
            self._check_model(project_root)

            # Lazy import to avoid pulling heavy deps at module import time
            from ..core.factory import ComponentFactory
            from ..core.search import SemanticSearchEngine

            bundle = await ComponentFactory.create_standard_components(
                project_root=project_root,
                include_search_engine=False,
            )
            await bundle.database.initialize()

            engine = SemanticSearchEngine(
                database=bundle.database,
                project_root=project_root,
                similarity_threshold=bundle.config.similarity_threshold,
            )

            lock = asyncio.Lock()
            self._cache[key] = (engine, lock)
            self._bundles[key] = bundle
            logger.info(f"Opened index for {key} (cache size={len(self._cache)})")

            # Evict LRU if over capacity
            while len(self._cache) > self.max_indexes:
                evicted_key, _ = self._cache.popitem(last=False)
                await self._close_one(evicted_key)
                logger.info(f"Evicted LRU index: {evicted_key}")

            return self._cache[key]

    async def _close_one(self, key: str) -> None:
        bundle = self._bundles.pop(key, None)
        if bundle is None:
            return
        try:
            await bundle.database.close()
        except Exception as e:  # pragma: no cover
            logger.warning(f"Error closing database for {key}: {e}")

    async def close_all(self) -> None:
        """Close every cached index. Called on daemon shutdown."""
        async with self._registry_lock:
            keys = list(self._cache.keys())
            self._cache.clear()
            for key in keys:
                await self._close_one(key)

    def open_indexes(self) -> list[str]:
        """Return the canonical paths of currently-open indexes (LRU order)."""
        return list(self._cache.keys())
