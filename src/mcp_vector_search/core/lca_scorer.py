"""Contrastive LCA (Lowest Common Ancestor) scoring for code hierarchies.

Adapted from the Nairobi Protocol GDE project's contrast_lca.py algorithm.
This module provides a "geometric resonance" score for any two nodes in a
CONTAINS hierarchy (module -> file -> class -> method) — useful as a
structural signal that complements vector similarity in KG-aware search
ranking.

Formula
-------
    penalty = |depth_a - lca_depth| + |depth_b - lca_depth|
              + 0.5 * |depth_a - depth_b|
    geometric_resonance = max(0.0, 1.0 - penalty / baseline)

Where ``baseline = 2 * max_depth`` of the CONTAINS tree, fixed at the time
the scorer is constructed (the hierarchy is static between KG builds).

Typical scores on a four-level (root/module/file/class/method) hierarchy:
    - same node                        -> 1.00
    - sibling methods (same class)     -> 0.75
    - methods in same file/diff class  -> 0.50
    - methods in same module/diff file -> 0.25
    - methods across modules           -> 0.00
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import networkx as nx
from loguru import logger

if TYPE_CHECKING:
    import kuzu


__all__ = [
    "ContrastiveLCAScorer",
    "build_lca_scorer_from_kuzu",
]


class ContrastiveLCAScorer:
    """Geometric-resonance scorer over a directed CONTAINS hierarchy.

    Wraps a ``networkx.DiGraph`` (parent -> child) and exposes ``score()``
    and ``score_query_vs_results()`` methods that return values in the
    closed range ``[0.0, 1.0]``.

    Args:
        graph: A ``networkx.DiGraph`` whose edges go parent -> child (CONTAINS
            direction).  The graph must be a tree (each node has at most one
            predecessor) for the LCA algorithm to be correct.
        root: The root node ID — typically a synthetic ``"root"`` node that
            connects to all top-level modules.
        baseline: Optional override for the normalization baseline.  When
            ``None``, the scorer uses ``2.0 * max_depth`` of the tree.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        root: str = "root",
        baseline: float | None = None,
    ) -> None:
        self._graph = graph
        self._root = root

        # Cache depths for every reachable node — the hierarchy is static
        # between builds so we compute these once.
        self._depths: dict[str, int] = {}
        if root in graph:
            try:
                self._depths = dict(nx.single_source_shortest_path_length(graph, root))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"LCA depth precompute failed: {exc}")
                self._depths = {}

        # Baseline = 2 * max_depth (worst case = two leaves whose LCA is root)
        if baseline is None:
            max_depth = max(self._depths.values()) if self._depths else 0
            self._baseline = 2.0 * max(1, max_depth)
        else:
            self._baseline = max(1.0, float(baseline))

        # LCA path cache (a, b) -> lca_id  (None means "no common ancestor")
        # Keyed in canonical (sorted) order so the cache is symmetric.
        self._lca_cache: dict[tuple[str, str], str | None] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def baseline(self) -> float:
        """Normalization baseline (== 2 * max_depth of the CONTAINS tree)."""
        return self._baseline

    @property
    def root(self) -> str:
        """Root node ID."""
        return self._root

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying CONTAINS DiGraph."""
        return self._graph

    def node_depth(self, node_id: str) -> int:
        """Return the depth of ``node_id`` from root (root = 0).

        Returns ``0`` for unknown nodes — callers should guard against this
        if they care about distinguishing "root" from "unknown".
        """
        return self._depths.get(node_id, 0)

    def score(self, node_a_id: str, node_b_id: str) -> float:
        """Return geometric resonance for ``(node_a_id, node_b_id)``.

        Returns ``0.0`` when either node is unknown (graceful fallback —
        the contract is "structural signal", so missing nodes contribute
        no signal rather than raising).
        """
        result = self.score_detailed(node_a_id, node_b_id)
        return float(result["geometric_resonance"])

    def score_detailed(self, node_a_id: str, node_b_id: str) -> dict[str, Any]:
        """Return the full LCA scoring detail for ``(node_a_id, node_b_id)``.

        The returned dict contains: ``lca``, ``depth_a``, ``depth_b``,
        ``lca_depth``, ``penalty``, ``baseline``, ``geometric_resonance``.
        """
        # Self-comparison shortcut
        if node_a_id == node_b_id:
            depth = self.node_depth(node_a_id)
            return {
                "node_a": node_a_id,
                "node_b": node_b_id,
                "lca": node_a_id if node_a_id in self._graph else None,
                "depth_a": depth,
                "depth_b": depth,
                "lca_depth": depth,
                "penalty": 0.0,
                "baseline": self._baseline,
                "geometric_resonance": (1.0 if node_a_id in self._graph else 0.0),
            }

        # Unknown node -> no signal
        if node_a_id not in self._graph or node_b_id not in self._graph:
            return {
                "node_a": node_a_id,
                "node_b": node_b_id,
                "lca": None,
                "depth_a": self.node_depth(node_a_id),
                "depth_b": self.node_depth(node_b_id),
                "lca_depth": 0,
                "penalty": float("inf"),
                "baseline": self._baseline,
                "geometric_resonance": 0.0,
            }

        lca = self._find_lca(node_a_id, node_b_id)
        if lca is None:
            return {
                "node_a": node_a_id,
                "node_b": node_b_id,
                "lca": None,
                "depth_a": self.node_depth(node_a_id),
                "depth_b": self.node_depth(node_b_id),
                "lca_depth": 0,
                "penalty": float("inf"),
                "baseline": self._baseline,
                "geometric_resonance": 0.0,
            }

        depth_a = self.node_depth(node_a_id)
        depth_b = self.node_depth(node_b_id)
        lca_depth = self.node_depth(lca)

        # Nairobi Protocol formula
        penalty = (
            abs(depth_a - lca_depth)
            + abs(depth_b - lca_depth)
            + 0.5 * abs(depth_a - depth_b)
        )
        resonance = max(0.0, 1.0 - penalty / self._baseline)

        return {
            "node_a": node_a_id,
            "node_b": node_b_id,
            "lca": lca,
            "depth_a": depth_a,
            "depth_b": depth_b,
            "lca_depth": lca_depth,
            "penalty": penalty,
            "baseline": self._baseline,
            "geometric_resonance": resonance,
        }

    def score_query_vs_results(
        self,
        query_node_id: str,
        result_node_ids: list[str],
    ) -> dict[str, float]:
        """Batch-score a query node against many candidate result nodes.

        Args:
            query_node_id: The "anchor" node (typically the entity matched
                by the search query).
            result_node_ids: Candidate result node IDs to score.

        Returns:
            ``{result_node_id: geometric_resonance}`` for each result node.
            Unknown result nodes map to ``0.0``.
        """
        return {rid: self.score(query_node_id, rid) for rid in result_node_ids}

    # ------------------------------------------------------------------
    # Internal LCA logic (cached)
    # ------------------------------------------------------------------

    def _find_lca(self, node_a: str, node_b: str) -> str | None:
        """Find the deepest common ancestor of ``node_a`` and ``node_b``.

        Uses the directed-tree predecessor walk: build root-first paths for
        each node and return the last shared element.
        """
        cache_key = (node_a, node_b) if node_a < node_b else (node_b, node_a)
        if cache_key in self._lca_cache:
            return self._lca_cache[cache_key]

        path_a = self._ancestors_with_self(node_a)
        path_b = self._ancestors_with_self(node_b)

        lca: str | None = None
        for a, b in zip(path_a, path_b, strict=False):
            if a == b:
                lca = a
            else:
                break

        self._lca_cache[cache_key] = lca
        return lca

    def _ancestors_with_self(self, node_id: str) -> list[str]:
        """Return the root-first list of ancestors of ``node_id`` (inclusive)."""
        path: list[str] = []
        current = node_id
        visited: set[str] = set()
        while True:
            if current in visited:
                break
            visited.add(current)
            path.append(current)
            preds = list(self._graph.predecessors(current))
            if not preds:
                break
            current = preds[0]  # tree -> at most one parent
        return list(reversed(path))


# ---------------------------------------------------------------------------
# Factory: build a scorer from a Kuzu connection
# ---------------------------------------------------------------------------


def build_lca_scorer_from_kuzu(
    connection: kuzu.Connection,
    root_id: str = "root",
) -> ContrastiveLCAScorer:
    """Build a ``ContrastiveLCAScorer`` from the live Kuzu KG.

    Queries every ``CONTAINS`` edge from the graph, materializes a
    ``networkx.DiGraph``, and synthesizes a top-level ``"root"`` node
    connected to any node that has no incoming CONTAINS edge.  This makes
    the LCA algorithm work even when the KG schema does not have a single
    real root entity.

    Args:
        connection: An open Kuzu connection (read-only is fine).
        root_id: The synthetic root node ID to use for top-level orphans.

    Returns:
        A fully-initialized ``ContrastiveLCAScorer``.
    """
    graph: nx.DiGraph = nx.DiGraph()

    try:
        result = connection.execute(
            """
            MATCH (a:CodeEntity)-[:CONTAINS]->(b:CodeEntity)
            RETURN a.id AS parent_id, b.id AS child_id
            """
        )
        # Kuzu's execute() may return list[QueryResult] for multi-statement
        # scripts. Normalize to a single result handle.
        if isinstance(result, list):
            result = result[0] if result else None

        edge_count = 0
        if result is not None:
            while result.has_next():
                row = cast(list[Any], result.get_next())
                parent_id, child_id = row[0], row[1]
                if parent_id is None or child_id is None:
                    continue
                graph.add_edge(parent_id, child_id, rel="CONTAINS")
                edge_count += 1

        logger.debug(
            f"build_lca_scorer_from_kuzu: loaded {edge_count} CONTAINS edges, "
            f"{graph.number_of_nodes()} nodes"
        )
    except Exception as exc:
        logger.warning(
            f"build_lca_scorer_from_kuzu: CONTAINS query failed ({exc}); "
            "returning empty scorer"
        )

    # Synthesize a root node connecting to all top-level orphans.
    if root_id not in graph:
        graph.add_node(root_id)

    orphans = [n for n in list(graph.nodes) if n != root_id and graph.in_degree(n) == 0]
    for orphan in orphans:
        graph.add_edge(root_id, orphan, rel="CONTAINS")

    return ContrastiveLCAScorer(graph, root=root_id)
