# Code Search Techniques: State of the Art
## Is Vector Similarity the Right Primary Technique? AST/KG Graph vs. Embeddings

**Date:** 2026-05-09
**Context:** mcp-vector-search uses CodeBERT/GraphCodeBERT embeddings + BM25 + RRF hybrid + KuzuDB knowledge graph
**Research Question:** Should we prioritize AST/KG graph search over vector similarity? What hybrid strategy is optimal?

---

## Executive Summary

Vector similarity is a strong but insufficient primary technique for code search. The research consensus (2024–2026) is that **no single technique dominates across all query types** — each has distinct failure modes. The optimal architecture is a layered hybrid: BM25 for exact identifiers, vector search for semantic intent, and graph traversal for structural/relational queries. The most significant near-term improvements for mcp-vector-search are: (1) upgrading from CodeBERT to CodeXEmbed or GraphCodeBERT, (2) implementing AST-aware chunking via tree-sitter, and (3) adding post-retrieval KG graph expansion to include callers/callees. These three changes address the most critical gaps relative to the current state of the art.

---

## 1. Current State of Code Search Techniques

### 1.1 The Technique Landscape

Four principal techniques exist, each with distinct trade-offs:

| Technique | Mechanism | Best For | Worst For |
|---|---|---|---|
| **Sparse/Keyword (BM25, TF-IDF)** | Token frequency statistics | Exact identifiers, API names, error codes | Natural language queries, synonyms |
| **Dense/Vector (CodeBERT, etc.)** | Neural embedding cosine similarity | Semantic intent, cross-language, concept search | OOV identifiers, long functions, cross-file reasoning |
| **Graph/Structural (AST, call graphs)** | Graph traversal + structural matching | "What calls X?", dependencies, type hierarchies | Natural language queries, concept search |
| **Structural Pattern (Comby, Semgrep, CodeQL)** | Syntax-tree pattern matching | Finding code patterns, vulnerability variants | Semantic queries, discovery scenarios |

### 1.2 Key Benchmarks

**CodeSearchNet** (2020, still used as baseline): 6 programming languages, 2M+ functions. Used for measuring NL-to-code retrieval (MRR metric). UniXcoder leads on this benchmark, followed by CodeT5, then GraphCodeBERT.

**CoIR** (2024, ACL 2025 Main): The most comprehensive modern benchmark with 10 datasets across 8 retrieval task types and 7 domains. Key findings:
- Even state-of-the-art retrievers perform "suboptimally" — APPS (code contest queries) averages NDCG@10 of only 11.25 across all models
- CosQA (web query to code) averages only 28.91 NDCG@10
- Top performer is **E5-Mistral** (general-purpose LLM-based embedder), not a code-specific model
- Voyage-Code-002 tops at 56.26 average NDCG@10 across all CoIR tasks
- Integrated into MTEB in August 2024; monthly downloads surpassed CodeSearchNet by September 2024
- Salesforce's **CodeXEmbed-7B** achieved rank #1 in early 2025, beating Voyage-Code-002 by over 20%
- Extending context from 512 to 4,096 tokens significantly improves performance (GTE: 28.48 → 51.32 on CodeFeedback-MT)

**Model Rankings on CodeSearchNet** (NL→code retrieval, MRR):
1. UniXcoder (uses AST + code comments for cross-modal alignment)
2. CodeT5 / CodeT5+
3. SynCoBERT
4. TreeBERT
5. GraphCodeBERT
6. PLBART

**CoIR Model Rankings** (average NDCG@10 across all 10 datasets):
1. CodeXEmbed-7B (Salesforce, 2025) — SOTA, +20% over Voyage-Code-002
2. Voyage-Code-002 — 56.26
3. E5-Mistral — 55.18
4. E5-Base — 50.90
5. OpenAI-Ada-002
6. UniXcoder, BGE-M3, GTE-Base, Contriever

### 1.3 Structural/Graph Tools Used in Industry

- **Sourcegraph**: Trigram-indexed keyword search + Comby structural search + LSIF/LSP precise code intelligence + optional vector semantic search. The combination is their recommended default.
- **GitHub Copilot** (2025): AST-chunked embeddings (tree-sitter), vector search via Turbopuffer/FAISS, plus symbol tables and "recently edited files" recency signal. New embedding model (2025) delivers +37.6% retrieval quality, 2x throughput, 8x memory reduction. Used Matryoshka Representation Learning (MRL) and contrastive training with hard negatives.
- **Cursor**: AST-based chunking via tree-sitter, embeddings stored in Turbopuffer, Merkle-tree-based incremental re-indexing. Exploring "multi-hop embedders" for chained dependency traversal.
- **Amazon Q Developer**: Creates a knowledge graph from the repository during documentation generation, combining graph traversal with semantic embeddings.
- **CodeQL**: Full semantic graph (AST + CFG + DFG + type hierarchies + call graphs). Declarative QL query language. Highest precision for structural queries; responsible for identifying 400+ CVEs. Not suited for discovery/exploration.

---

## 2. Vector Search Limitations for Code

### 2.1 Failure Mode 1: Out-of-Vocabulary (OOV) Identifiers

CodeBERT and most transformer-based models tokenize code with a fixed vocabulary (32K–50K subword tokens). Internal identifiers like `getOAuthTokenFromKeycloak`, `PROD-SKU-7842X`, or `handleUserSessionExpiry` are split into subword fragments that lose semantic coherence.

**Example:** A vector search for `"getOAuthTokenFromKeycloak"` may rank `"handleUserLogin"` higher than the actual function because the subword fragments `["get", "OAuth", "Token", "From", "Key", "cloak"]` produce an embedding that resembles authentication-related vocabulary generally, while the model has never seen "Keycloak" as a coherent unit.

Dense vector models "sacrifice exact string matching" to achieve semantic generalization — this is structurally unavoidable. BM25 handles OOV identifiers correctly because it treats tokens as exact strings.

Real-world evaluation of GraphCodeBERT on a Python repository (193 functions, Elasticsearch): "The model correctly identifies the function only when the query is very specific and closely matches the original wording. When queries are slightly modified or synonyms are used, the results seem almost random."

### 2.2 Failure Mode 2: Context Window Truncation

CodeBERT's maximum input is 512 tokens. A typical Python function with docstring, decorators, complex logic, and proper type annotations can easily exceed this. When truncated:
- The embedding represents only the first 512 tokens (usually the signature and early body)
- Key logic at the end of the function is invisible to search
- Long classes are almost always truncated

**Measured impact (CoIR):** Extending context from 512 to 4,096 tokens improves NDCG@10 from 28.48 to 51.32 on CodeFeedback-MT — an 80% improvement just from longer context. This is the single most impactful technical limitation of CodeBERT for mcp-vector-search.

### 2.3 Failure Mode 3: Semantic Drift / False Positive Retrieval

Sourcegraph identifies this directly: searching for "the function called processOrder" may incorrectly surface `handlePurchase` and `completeTransaction` because they share authentication/transaction vocabulary.

Domain-specific terminology also causes drift: a codebase with custom internal terminology (e.g., "nairobi protocol," "GDE analysis") confuses models trained on public GitHub code.

Semantic drift worsens in multi-turn agent sessions as context accumulates and the original query intent loses salience relative to accumulated noise.

### 2.4 Failure Mode 4: Cross-File Reasoning

Vector search fundamentally cannot answer:
- "What calls this function?" → requires call graph traversal
- "What depends on this module?" → requires dependency graph traversal
- "What is the full call chain from the API handler to the database?" → requires multi-hop traversal
- "If I change this function signature, what breaks?" → requires reverse dependency lookup

File-level vector embeddings can identify "semantically similar files" but cannot identify "genuinely related code" — the distinction RepoGraph (ASE 2024) emphasizes. Vector similarity is topology-blind.

### 2.5 Failure Mode 5: Chunking Artifacts

Current mcp-vector-search chunking strategy (inferred from `chunk_processor.py`): function/method-level with class context injection. This is already above baseline, but:

- If functions exceed 512 tokens (CodeBERT limit), they are truncated
- Cross-function context (what does this function call within the same file?) is lost at chunk boundaries
- Class-level relationships beyond simple parent-child are not preserved in chunk embeddings

Naive fixed-size chunking "can break the structure of a method, causing the model to lose context regarding its return value" (cAST paper, CMU/Augment Code, EMNLP 2025). The cAST paper shows AST-aware chunking improves Recall@5 by 4.3 points and Pass@1 by 2.67 points on SWE-bench vs. line-based chunking.

---

## 3. Graph/AST Search Advantages

### 3.1 What Graph Search Does Uniquely Well

Graph traversal answers queries that are structurally impossible for vector similarity:

**Exact relational queries:**
- "Find all functions that call `authenticate()`" → CALLS traversal, O(1) KG lookup
- "What modules import `utils.crypto`?" → IMPORTS traversal, O(1) KG lookup
- "Find all subclasses of `BaseSerializer`" → INHERITS traversal
- "What is the full dependency chain for `OrderService`?" → multi-hop IMPORTS/CALLS

**Change impact analysis:**
- "If I change `parse_config()`, what code paths are affected?" → reverse CALLS traversal
- "Which tests cover this function?" → CALLS chain from test functions

**Type-aware structural queries:**
- "Find all implementations of interface `IRepository`" → type hierarchy traversal
- "Find all places where `UserModel` is instantiated" → reference graph traversal

These queries return 100% precision answers — not probability-ranked candidates. No vector model can match this for structural queries.

### 3.2 GraphCodeBERT: Structure-Aware Embeddings

GraphCodeBERT (Microsoft, ICLR 2021) encodes **data flow graphs** — "where-the-value-comes-from" relationships between variables — directly into the embedding space. Unlike pure token-sequence models:

- Uses semantic-level structure (data flow) rather than syntactic structure (AST), avoiding "unnecessarily deep hierarchy"
- Introduces graph-guided masked attention to incorporate code structure into transformer layers
- Two additional pretraining tasks: predict code structure edges + align source code with structure
- Ranks 5th on CodeSearchNet (vs. CodeBERT at lower rank), but is more structurally grounded

**Limitation:** Even GraphCodeBERT is still a vector model — it encodes graph structure into a fixed-size embedding, losing the ability to do arbitrary graph traversal at query time.

### 3.3 GraphCoder: Graph Retrieval at Query Time (ASE 2024)

GraphCoder (Liu et al., ASE 2024) represents the most relevant recent graph-first approach for code completion:

**Architecture:**
1. Builds a Code Context Graph (CCG) from code using tree-sitter ASTs: control flow edges, control dependence edges, data dependence edges
2. Coarse retrieval: Jaccard similarity on bag-of-words to find candidate snippets
3. Fine-grained re-ranking: "decay-with-distance subgraph edit distance" — structural graph similarity weighted by distance from the completion target

**Results (8,000 completion tasks, 20 repositories, 5 LLMs):**
- +6.06 improvement in code match exact match vs. baseline RAG
- +6.23 improvement in identifier match EM vs. baseline RAG
- +7.90 for line-level tasks, +4.58 for API-level tasks specifically vs. vanilla RAG
- Achieves this with **less time and storage** than vanilla RAG

### 3.4 RepoGraph: Line-Level Graph Retrieval (arXiv 2024)

RepoGraph builds a directed graph where nodes are individual code lines (def or ref nodes) with invoke edges (call relationships) and contain edges (nesting relationships). Applied as a post-processing layer on top of existing frameworks:

| Framework | Baseline | +RepoGraph | Improvement |
|---|---|---|---|
| RAG (GPT-4) | 2.67% | 5.33% | +100% relative |
| Agentless (GPT-4o) | 27.33% | 29.67% | +8.6% |
| AutoCodeRover (GPT-4) | 19.0% | 21.33% | +12.3% |
| SWE-agent (GPT-4o) | 18.33% | 20.33% | +10.9% |

**Average relative improvement: +32.8%** — purely by adding graph context on top of existing retrieval, with no changes to the underlying approach.

Key insight from the paper: "indexing at file-level can only identify semantically similar but not genuinely related code snippets." Graph traversal captures actual dependency relationships rather than superficial similarity.

### 3.5 Knowledge Graph for Repository Code Generation (ICSE LLM4Code 2025)

KG-based approach (Athale & Vaddina, 2025) using Neo4j with nodes (File, Class, Method, Function, Attribute, GeneratedDescription) and edges (defines_class, defines_function, has_method, used_in, has_attribute, has_description):

- N-hop graph traversal for dependency and usage pattern discovery after initial semantic retrieval
- Sub-graph semantic ranking: embed each KG node, filter by similarity to query, pass top-k to LLM
- Result: 36.36% pass@1 with Claude 3.5 Sonnet on EvoCodeBench vs. 20.73% for local file infilling, 17.45% for local file completion, 7.27% for no context — roughly 2x over the strongest baseline
- Comparable to CodeXGraph (36.02% on GPT-4o) but evaluated on full 275-sample set vs. CodeXGraph's 212

### 3.6 SemanticForge: Semantic Knowledge Graphs (arXiv 2025)

On RepoKG-50 (4,250 tasks, 50 Python projects):
- 49.8% Pass@1 (+15.6 absolute points over base Code-Llama-34B)
- 49.8% reduction in schematic hallucination via SMT-guided generation
- 34.7% reduction in logical hallucination via dual graph analysis
- Sub-3 second latency via incremental algorithms

---

## 4. Best Hybrid Strategy: What Leading Tools Do

### 4.1 The Convergent Industry Architecture

All major code AI tools (Copilot, Cursor, Sourcegraph, Amazon Q) converge on the same pattern:

```
Query
  |
  +---> [1] Lexical/BM25 ---------> Candidates A (exact match, identifiers)
  |
  +---> [2] Vector/Semantic -------> Candidates B (intent, concepts)
  |
  +---> [3] Graph/Structural ------> Candidates C (relationships, dependencies)
            (optional, query-type triggered)
  |
  +---> RRF Fusion / Re-ranking ---> Merged ranked list
  |
  +---> [4] Graph Expansion -------> Expand with callers/callees/dependencies
  |
  +---> LLM with retrieved context
```

**Sourcegraph's model:** "Use keyword search when you know what you're looking for, structural search when you know the code pattern, and semantic search when you know the concept but not the implementation." Each technique is routed based on query characteristics.

**GitHub Copilot's model:** AST-chunked embeddings + symbol tables + recently-edited-files recency. The 2025 embedding model uses MRL (Matryoshka Representation Learning) to handle both small fragments and entire files. Hard negatives (functionally different but superficially similar code) during training dramatically reduce false positives.

**Cursor's future direction:** Exploring "multi-hop embedders" — given a query and the relevant code found so far, the model determines the next piece of code to hop to. This directly addresses the cross-file reasoning gap.

### 4.2 GraphRAG for Code: The Post-Retrieval Expansion Pattern

The GraphRAG pattern (Microsoft, 2024) adapted for code:

1. **Initial retrieval**: vector + BM25 → top-k candidates
2. **Graph expansion**: from retrieved nodes, traverse KG edges (CALLS, IMPORTS, CONTAINS) to include related nodes within N hops
3. **Semantic pruning**: embed expanded subgraph nodes, filter by similarity to query, keep top-k
4. **Re-ranking**: re-rank merged set using cross-encoder or structural similarity
5. **LLM generation**: use pruned, expanded context

LightRAG (October 2024) achieves comparable accuracy to full GraphRAG with 10x token reduction through dual-level retrieval. The GAHR-MSR framework (ICNLSP 2024) shows +25.4% nDCG@10 improvement (0.685 → 0.859) and +13.3% Recall@100 over dense-only baseline using graph metadata filtering + ColBERT re-ranking.

### 4.3 What Research Recommends Specifically for Code

From the CoIR paper (ACL 2025): "code data is semi-structured and inherently logical, requiring specialized approaches." Key recommendations:
- Code-specific pretraining matters significantly
- Input length: extending from 512 to 4,096+ tokens is the highest-leverage technical improvement
- No single model dominates all retrieval task types — suggest task-aware routing

From cAST (CMU/Augment Code, EMNLP 2025): AST-aware chunking outperforms line-based chunking with +4.3 Recall@5 and +2.67 Pass@1 improvement. The ASTChunk library uses tree-sitter, supports Python/Java/C#/TypeScript.

From GraphCoder (ASE 2024): Graph-based re-ranking after initial coarse retrieval consistently outperforms sequence-based RAG baselines with lower compute overhead.

From RepoGraph (arXiv 2024): Adding graph expansion on top of existing RAG gives average +32.8% relative improvement with no changes to the retrieval model itself.

---

## 5. Concrete Recommendations for mcp-vector-search

The following recommendations are prioritized by expected impact and implementation effort, grounded in the research above.

### Priority 1: Upgrade the Embedding Model (High Impact, Medium Effort)

**Current:** CodeBERT (768-dim, 512 token limit)
**Recommended:** GraphCodeBERT or CodeXEmbed-400M

**Why:**
- CodeBERT's 512-token limit is the biggest measurable gap relative to current SOTA. The CoIR paper demonstrates an 80% NDCG@10 improvement just from extending context to 4,096 tokens.
- GraphCodeBERT already incorporates data flow graph structure into pretraining, directly addressing the "structure-blind embedding" problem without changing retrieval architecture.
- CodeXEmbed-7B achieves +20% over Voyage-Code-002 on CoIR. The 400M variant still significantly outperforms Voyage-Code-002 at a deployable size.
- mcp-vector-search already has a migration path (`migrations/v1_2_2_codexembed.py` exists), suggesting this was considered.

**Action:** Evaluate GraphCodeBERT (768-dim, 512 tokens but structure-aware) vs. CodeXEmbed-400M (longer context, multilingual, CoIR SOTA). CodeXEmbed-400M is likely the better choice for production due to CoIR benchmark evidence and multi-language support.

**Note on current CodeBERT vs GraphCodeBERT already in use:** The codebase references GraphCodeBERT in `embeddings.py` (768-dim) and the migration file references CodeXEmbed. Confirm which model is active in production and validate against CoIR task types that match mcp-vector-search query patterns.

### Priority 2: AST-Aware Chunking via tree-sitter (High Impact, Medium Effort)

**Current:** Function/method-level chunking (already better than file-level or fixed-size)
**Recommended:** AST-structural chunking following cAST methodology

**Why:**
- Function-level chunking is already sound for most cases
- The remaining gap: long functions exceeding 512 tokens (CodeBERT) or 4,096 tokens (CodeXEmbed) are still truncated
- Functions with complex nested structures (list comprehensions, generators, lambda chains) may cross chunk boundaries in ways that break semantic coherence
- cAST (EMNLP 2025) demonstrates +4.3 Recall@5 improvement over line-based chunking using AST-aware boundaries
- The ASTChunk library (tree-sitter-based) is directly usable: `pip install astchunk`

**Implementation:**
- Replace or augment current chunking with AST-structural splitting using tree-sitter
- For functions/methods exceeding token limit: split at statement-level AST boundaries rather than character count
- Inject parent context (class definition, imports) into each chunk as a prefix header (already partially done based on `chunk_processor.py` evidence)
- Ensure chunk metadata preserves: file path, class name, function name, start/end lines, language

### Priority 3: Post-Retrieval KG Graph Expansion (High Impact, Medium Effort)

**Current:** KG is built (CALLS, IMPORTS, CONTAINS, DOCUMENTS relationships) but appears used primarily for analysis tools (`kg_query`, `kg_history`, `trace_execution_flow`) rather than as a live retrieval enrichment signal during search.

**Recommended:** Graph expansion as a post-retrieval step in `search_handler.py`

**Why:**
- RepoGraph shows +32.8% average relative improvement on SWE-bench by adding graph context to existing RAG — no model changes required
- The KG already exists with CALLS and IMPORTS edges; this is a software engineering problem, not a model training problem
- Directly addresses the cross-file reasoning gap: after retrieving function F via vector search, expand to include: callers of F (1-hop CALLS), functions F calls (1-hop CALLS), modules F imports (1-hop IMPORTS), sibling methods in same class (CONTAINS siblings)

**Implementation Pattern:**
```python
# After initial vector/BM25 retrieval returns top-k chunk_ids:
async def expand_with_graph(chunk_ids: list[str], hops: int = 1) -> list[str]:
    """Expand retrieval set by traversing KG edges from initial results."""
    expanded = set(chunk_ids)
    for chunk_id in chunk_ids:
        # 1-hop CALLS expansion (what this function calls + what calls this function)
        callers = await kg.get_callers(chunk_id)
        callees = await kg.get_callees(chunk_id)
        expanded.update(callers[:2])  # limit: top-2 callers
        expanded.update(callees[:2])  # limit: top-2 callees
        # 1-hop IMPORTS expansion (what this module imports)
        imports = await kg.get_imports(chunk_id)
        expanded.update(imports[:3])
    return list(expanded)
```

**Graph Pruning:** After expansion, embed expanded nodes and filter by cosine similarity to query (keep only nodes with similarity > threshold). This prevents graph explosion from hub nodes. The KG-based repo code generation paper (ICSE 2025) describes this exact pattern: "sub-graph is refined through semantic ranking, prioritizing nodes that align most closely with the query's purpose."

**Where to integrate:** In `search_handler.py` or `search.py`, after the RRF fusion step and before the cross-encoder reranker. The expanded candidates feed into `CrossEncoderReranker` which already exists in `reranker.py`.

### Priority 4: Query-Type-Aware Routing (Medium Impact, Low Effort)

**Current:** `search.py` already has one routing heuristic: auto-adjusts `hybrid_alpha` for identifier-style queries (shifts toward BM25 when query looks like an SDK name/package name). This is the right approach.

**Recommended:** Extend routing logic to detect when graph traversal is the correct primary technique:

```python
GRAPH_QUERY_PATTERNS = [
    r"what calls? .+\??$",
    r"what (depends|imports|uses) .+\??$",
    r"callers? of .+$",
    r"where is .+ (called|used|imported)\??$",
    r"impact of (changing|modifying) .+$",
    r"what breaks? if .+$",
]
```

When a query matches graph patterns, query the KG directly first (using `knowledge_graph.py` Cypher queries), then use vector search only for semantic enrichment of the results.

This directly addresses the "Sourcegraph principle": route to the right tool based on query type rather than always running all retrievers.

### Priority 5: Structural Pattern Search as Pre-Filter (Lower Priority, Higher Effort)

**Recommended:** Add Comby or Semgrep integration as an optional retrieval path for structural queries.

**Why lower priority:** The KG already answers most structural questions that Comby/Semgrep would handle (CALLS relationships subsume "find all callers"), and the implementation cost is higher. However, for queries like "find all try/except blocks that catch `Exception` broadly" or "find all functions with more than 5 parameters," neither vector search nor the current KG can answer these — they require AST pattern matching.

**When to prioritize:** If user feedback reveals common query patterns about code structure (anti-pattern detection, style searches) that neither vector nor KG retrieval handles well.

### Priority 6: Cross-Encoder Re-ranking (Already Present — Tune It)

The `reranker.py` with `CrossEncoderReranker` already exists. Cross-encoder re-ranking (the final ColBERT-style or cross-encoder scoring step) is the right architecture for re-ranking a merged candidate set.

**Immediate tuning:** Ensure the cross-encoder is seeing the graph-expanded candidates (from Priority 3), not just the initial vector/BM25 results. The reranker's `rerank_top_n=50` parameter should be increased if graph expansion adds candidates — consider 100+ when graph expansion is active.

---

## 6. Synthesis: The Optimal Architecture for mcp-vector-search

```
User Query
    |
    v
[Query Classification]
    |
    +-- Identifier query (e.g., "getOAuthToken")
    |       --> BM25 primary (alpha=0.1), skip graph expansion
    |
    +-- Structural query (e.g., "what calls authenticate()")
    |       --> KG direct traversal primary
    |           --> vector enrichment secondary
    |
    +-- Semantic query (e.g., "how is user authentication handled")
            --> Current hybrid RRF (alpha=0.7)
            --> + KG expansion (Priority 3)
            --> + Cross-encoder reranker (already present)
                |
                v
        [AST-chunked embeddings] (Priority 2)
        using [GraphCodeBERT or CodeXEmbed-400M] (Priority 1)
```

**Expected cumulative improvement over current baseline:**
- Embedding upgrade (CodeBERT → CodeXEmbed-400M): estimated +15-25% NDCG@10 based on CoIR benchmarks
- AST chunking: +4.3 Recall@5 (cAST results)
- KG graph expansion: +32.8% relative on structural tasks (RepoGraph results)
- Query routing (existing alpha adjustment already contributes): +10-15% on identifier queries

The key insight from the literature: **vector search and graph search are complementary, not competing.** Vector search finds semantically similar code efficiently. Graph search finds structurally related code precisely. The optimal system uses vector search for retrieval and graph traversal for expansion. The mcp-vector-search architecture already has both components; the gap is in connecting them during the search path.

---

## References

- [CoIR Benchmark (Li et al., ACL 2025)](https://arxiv.org/abs/2407.02883) — Comprehensive code retrieval benchmark; CoIR GitHub
- [GraphCoder (Liu et al., ASE 2024)](https://arxiv.org/abs/2406.07003) — Graph-based coarse-to-fine code completion retrieval
- [RepoGraph (arXiv 2024)](https://arxiv.org/html/2410.14684v1) — Line-level call graph for repository-level code tasks
- [cAST (Zhang et al., EMNLP 2025)](https://arxiv.org/abs/2506.15655) — AST-structural chunking for code RAG
- [GraphCodeBERT (Guo et al., ICLR 2021)](https://arxiv.org/abs/2009.08366) — Data flow graph pretraining for code representation
- [CodeXEmbed (Salesforce, COLM 2025)](https://arxiv.org/abs/2411.12644) — SOTA code embedding family, CoIR rank #1
- [KG-Based Repo Code Generation (Athale & Vaddina, ICSE LLM4Code 2025)](https://arxiv.org/abs/2505.14394) — Neo4j KG + N-hop expansion for code generation
- [SemanticForge (arXiv 2025)](https://arxiv.org/html/2511.07584) — Semantic KG + SMT-guided generation for code
- [GitHub Copilot Embedding Model (GitHub Blog, 2025)](https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/) — MRL training, +37.6% retrieval quality
- [Sourcegraph Semantic Code Search](https://sourcegraph.com/blog/semantic-code-search-what-it-is-and-how-it-works) — Hybrid keyword + structural + semantic approach
- [How Cursor Indexes Codebases](https://read.engineerscodex.com/p/how-cursor-indexes-codebases-fast) — AST chunking + Turbopuffer + Merkle diffing
- [Graph-Based Re-ranking Survey (arXiv 2503.14802, 2025)](https://arxiv.org/html/2503.14802v1) — Two-phase retrieval with graph expansion
- [GAHR-MSR Framework (ICNLSP 2024)](https://dev.to/lucash_ribeiro_dev/graph-augmented-hybrid-retrieval-and-multi-stage-re-ranking-a-framework-for-high-fidelity-chunk-50ca) — +25.4% nDCG@10 with graph metadata + ColBERT re-ranking
- [GraphRAG Survey (Peng et al., 2024)](https://arxiv.org/pdf/2408.08921) — Graph-based indexing, retrieval, and generation survey
- [About CodeQL](https://codeql.github.com/docs/codeql-overview/about-codeql/) — Semantic code analysis as relational graph queries
- [GraphRAG for Code (Knowledge Graph Based Repo-Level Code Generation)](https://arxiv.org/html/2505.14394v1)
- [LoRACode: Fine-tuning Code Embeddings (arXiv 2503.05315)](https://arxiv.org/html/2503.05315v1) — +14.8% MRR, +13.5% NDCG with LoRA fine-tuning
- [CoSQA+ Enhanced Code Search Dataset (arXiv 2406.11589)](https://arxiv.org/html/2406.11589v1)
- [Code Similarity Using GNNs (Stanford CS224W)](https://medium.com/stanford-cs224w/code-similarity-using-graph-neural-networks-1e58aa21bd92)
- [Retrieval-Augmented Code Generation Survey (arXiv 2510.04905)](https://arxiv.org/html/2510.04905v1)
