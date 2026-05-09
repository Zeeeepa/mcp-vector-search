"""Isolated subprocess for KG building - NO LANCEDB IMPORTS!

This module is intentionally isolated from the main codebase to prevent
any LanceDB background threads from being created. It's invoked as a
completely separate Python process.
"""

import json
import os
import sys
import threading
from pathlib import Path

# CRITICAL: Set environment variables BEFORE any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Entry point for subprocess KG build."""
    import argparse
    import sys

    from loguru import logger
    from rich.console import Console
    from rich.table import Table

    from mcp_vector_search.core.kg_builder import KGBuilder
    from mcp_vector_search.core.knowledge_graph import KnowledgeGraph
    from mcp_vector_search.core.models import CodeChunk
    from mcp_vector_search.core.progress import ProgressTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=str)
    parser.add_argument("chunks_file", type=str)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-documents", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debug output"
    )
    parser.add_argument(
        "--files-to-delete",
        type=str,
        default=None,
        help="Path to JSON file listing file paths whose KG entities should be deleted "
        "before building (used by incremental builds)",
    )
    args = parser.parse_args()

    # Suppress DEBUG logs unless --verbose
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    console = Console()
    project_root = Path(args.project_root)
    chunks_file = Path(args.chunks_file)

    # Check threads immediately
    threads = threading.enumerate()
    if args.verbose:
        console.print(f"[cyan]🔍 Thread check: {len(threads)} thread(s) active[/cyan]")
        for t in threads:
            console.print(f"  - {t.name} (daemon={t.daemon})")

    if len(threads) > 1:
        background_threads = [t for t in threads if t != threading.main_thread()]
        if background_threads:
            console.print(
                f"[red]✗ ERROR: {len(background_threads)} background thread(s) detected![/red]"
            )
            console.print(
                "[red]Kuzu requires single-threaded execution. Background threads (even daemons) "
                "cause segfaults during relationship insertion.[/red]"
            )
            for t in background_threads:
                console.print(f"  - {t.name} (daemon={t.daemon})")
            return 1

    try:
        # Load chunks from JSON file (no database access!)
        if args.verbose:
            console.print(f"[cyan]Loading chunks from {chunks_file.name}...[/cyan]")
        with open(chunks_file) as f:
            chunks_data = json.load(f)

        # Deserialize chunks
        # NOTE: chunks_data comes from chunks.lance which has different field names
        # than CodeChunk. We need to map/filter fields appropriately.
        chunks = []
        for chunk_dict in chunks_data:
            # Convert file_path to Path object
            chunk_dict["file_path"] = Path(chunk_dict["file_path"])

            # Map chunks.lance fields to CodeChunk fields
            # chunks.lance has "name" but CodeChunk expects function_name/class_name
            if "name" in chunk_dict:
                name = chunk_dict.pop("name")
                # Set function_name or class_name based on chunk_type
                if chunk_dict.get("chunk_type") == "class":
                    chunk_dict["class_name"] = name
                else:
                    chunk_dict["function_name"] = name

            # Map complexity -> complexity_score
            if "complexity" in chunk_dict:
                chunk_dict["complexity_score"] = chunk_dict.pop("complexity")

            # Remove fields that CodeChunk doesn't have
            extra_fields = [
                "file_hash",  # Change detection only (not in CodeChunk)
                "start_char",
                "end_char",
                "parent_name",  # chunks.lance uses parent_name, CodeChunk uses parent_chunk_id
                "hierarchy_path",
                "signature",
                "token_count",
                "embedding_status",
                "embedding_batch_id",
                "created_at",
                "updated_at",
                "error_message",
            ]
            for field in extra_fields:
                chunk_dict.pop(field, None)

            chunks.append(CodeChunk(**chunk_dict))

        if args.verbose:
            console.print(f"[green]✓ Loaded {len(chunks)} chunks[/green]")

        # Memory guard: warn loudly when a large KG build is starting on a
        # host that may not have enough RAM.  The full guard logic lives in
        # kg_builder._check_kg_memory_guard; we approximate file_count here
        # from chunks so we can warn before any KG work begins.
        try:
            from mcp_vector_search.core.kg_builder import _check_kg_memory_guard

            file_count = len({str(c.file_path) for c in chunks if c.file_path})
            _check_kg_memory_guard(file_count, console)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Pre-build memory guard check failed: %s", e)

        if len(chunks) == 0:
            console.print(
                "[red]✗[/red] No chunks found. Run 'mcp-vector-search index' first."
            )
            return 1

        # Initialize knowledge graph
        # Check for custom KG path suffix (set by index command for atomic rebuild)
        kg_path_suffix = os.environ.get("KG_PATH_SUFFIX", "knowledge_graph")
        kg_path = project_root / ".mcp-vector-search" / kg_path_suffix

        # Force rebuild if requested
        if args.force and kg_path.exists():
            if args.verbose:
                console.print(
                    "[yellow]🗑️  Force rebuild: removing existing KG...[/yellow]"
                )
            import shutil

            for item in kg_path.iterdir():
                if item.name.startswith("code_kg"):
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            if args.verbose:
                console.print("[green]✓ Old KG files removed[/green]")

        kg = KnowledgeGraph(kg_path)
        kg.initialize_sync()

        # --- Incremental: delete stale entities BEFORE building new ones ---
        if args.files_to_delete and Path(args.files_to_delete).exists():
            with open(args.files_to_delete) as f:
                files_to_delete = json.load(f)
            if files_to_delete:
                if args.verbose:
                    console.print(
                        f"[yellow]Deleting KG entities for {len(files_to_delete)} "
                        f"changed/removed file(s)...[/yellow]"
                    )
                kg.delete_entities_for_files(files_to_delete)
                if args.verbose:
                    console.print(
                        f"[green]✓ Deleted entities for {len(files_to_delete)} file(s)[/green]"
                    )
            logger.info(
                "Incremental KG: deleted entities for %d file(s)", len(files_to_delete)
            )

        # Determine whether the parent invoked us in "diff mode".  Diff mode is
        # signaled by --files-to-delete pointing to a real file (it may be
        # empty, meaning "no changes, just confirm up-to-date status").
        diff_mode = bool(args.files_to_delete) and Path(args.files_to_delete).exists()
        files_to_delete_count = 0
        if diff_mode:
            try:
                with open(args.files_to_delete) as _fdel:
                    _fdel_data = json.load(_fdel)
                    files_to_delete_count = (
                        len(_fdel_data) if isinstance(_fdel_data, list) else 0
                    )
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Failed to recount files-to-delete file: %s", e)

        # In diff mode, if BOTH the delete-list and the chunks list are empty,
        # there is nothing to do — the KG is already up to date.  Print a clear
        # "up to date" message and exit 0.
        if diff_mode and files_to_delete_count == 0 and len(chunks) == 0:
            stats = kg.get_stats_sync()
            console.print(
                f"[green]✓[/green] Knowledge graph is up to date "
                f"({stats['total_entities']} entities)."
            )
            kg.close_sync()
            return 0

        # Check if graph already exists — skip this guard in diff/incremental mode
        # (a non-empty diff_mode indicates we are *updating* the graph, not
        # replacing it, so a non-empty existing graph is expected)
        if not args.force and not diff_mode:
            stats = kg.get_stats_sync()
            if stats["total_entities"] > 0:
                console.print(
                    f"[yellow]⚠[/yellow] Knowledge graph already exists "
                    f"({stats['total_entities']} entities)."
                )
                console.print(
                    "  Run [cyan]'mcp-vector-search kg status'[/cyan] to view details."
                )
                console.print("  Use [cyan]--force[/cyan] to rebuild.")
                kg.close_sync()
                return 0

        # Build graph with smaller batch size to avoid Kuzu segfault
        # Kuzu can segfault with large batches (17k relationships)
        builder = KGBuilder(kg, project_root)

        # Override batch size for relationship insertion
        safe_batch_size = 100  # Smaller batches to avoid segfaults
        if args.verbose:
            console.print(
                f"[dim]Using batch size: {safe_batch_size} for relationship insertion[/dim]"
            )

        # Create progress tracker
        progress_tracker = ProgressTracker(console, verbose=args.verbose)

        build_stats = builder.build_from_chunks_sync(
            chunks,
            show_progress=True,
            skip_documents=args.skip_documents,
            progress_tracker=progress_tracker,
        )

        # Show results
        table = Table(title="Knowledge Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Code Entities", str(build_stats["entities"]))
        table.add_row("Doc Sections", str(build_stats.get("doc_sections", 0)))
        table.add_row("Documents", str(build_stats.get("doc_nodes", 0)))
        table.add_row("Tags", str(build_stats.get("tags", 0)))
        table.add_row("Persons", str(build_stats.get("persons", 0)))
        table.add_row("Projects", str(build_stats.get("projects", 0)))
        table.add_row("Calls", str(build_stats["calls"]))
        table.add_row("Imports", str(build_stats["imports"]))
        table.add_row("Inherits", str(build_stats["inherits"]))
        table.add_row("Contains", str(build_stats["contains"]))
        table.add_row("References", str(build_stats.get("references", 0)))
        table.add_row("Follows", str(build_stats.get("follows", 0)))
        table.add_row("Has Tag", str(build_stats.get("has_tag", 0)))
        table.add_row("Demonstrates", str(build_stats.get("demonstrates", 0)))
        table.add_row("Links To", str(build_stats.get("links_to", 0)))
        table.add_row("Authored", str(build_stats.get("authored", 0)))
        table.add_row("Modified", str(build_stats.get("modified", 0)))
        table.add_row("Part Of", str(build_stats.get("part_of", 0)))

        console.print(table)
        console.print("[green]✓[/green] Knowledge graph built successfully!")

        kg.close_sync()
        return 0

    except Exception as e:
        console.print(f"[red]✗ Build failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Clean up temp file
        try:
            chunks_file.unlink()
        except Exception as e:
            logger.debug("Failed to clean up temp file %s: %s", chunks_file, e)


if __name__ == "__main__":
    sys.exit(main())
