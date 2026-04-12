"""chat-local command: LLM-powered code chat using local Ollama inference."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.columns import Columns
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ...core.config_utils import is_bedrock_available
from ...core.embeddings import create_embedding_function, suppress_stdout_stderr
from ...core.exceptions import ProjectNotFoundError
from ...core.factory import create_database
from ...core.llm_client import LLMClient
from ...core.ollama_detector import detect_best_model, detect_ollama, get_model_limits
from ...core.project import ProjectManager
from ...core.search import SemanticSearchEngine
from ..didyoumean import create_enhanced_typer
from ..output import console, print_error, print_warning
from .chat import (
    EnhancedChatSession,
    _execute_tool,
    _get_tools,
)

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

chat_local_app = create_enhanced_typer(
    help="Chat with your codebase using local Ollama inference (no API key needed)",
    invoke_without_command=True,
    no_args_is_help=False,
)

# ---------------------------------------------------------------------------
# System prompt tuned for local/Gemma models
# (shorter, more direct — small-context models work better with concise prompts)
# ---------------------------------------------------------------------------

LOCAL_SYSTEM_PROMPT = """You are a concise code assistant. Use the tools to search code and answer questions.

RULES:
1. Be brief and direct. Answer in 2-4 sentences when possible.
2. Use search_code to find relevant code before answering.
3. Only show code when explicitly requested.
4. Reference file names and function names from search results.
5. After finding relevant code, give your final answer directly.

TOOL USAGE: Use search_code first for any code question. Use read_file only when you need the full file."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_ollama_not_running() -> None:
    message = """[bold yellow]Ollama Not Running[/bold yellow]

[bold cyan]To use local inference:[/bold cyan]
  1. Install Ollama: [cyan]https://ollama.ai[/cyan]
  2. Start the server: [yellow]ollama serve[/yellow]
  3. Pull a model:    [yellow]ollama pull gemma3[/yellow]
  4. Run again:       [yellow]mvs chat-local[/yellow]

[dim]Alternatively, use the cloud-based chat command:[/dim]
  [yellow]mvs chat[/yellow]"""

    console.print(Panel(message, border_style="yellow", padding=(1, 2)))


def _show_intro(model_name: str) -> None:
    intro = f"""[bold cyan]MCP Vector Search - Local Chat[/bold cyan]

[dim]Using local inference — no API key required.[/dim]

[bold]Model:[/bold] [green]{model_name}[/green]
[bold]Provider:[/bold] Ollama (local)

[bold]Commands:[/bold] /task, /status, /clear, /exit"""

    console.print(Panel(intro, border_style="cyan", padding=(1, 2)))


# ---------------------------------------------------------------------------
# JSON output for machine consumption (--json flag)
# ---------------------------------------------------------------------------


def _print_json_output(
    model: str,
    question: str,
    answer: str,
    search_results: list[dict[str, Any]],
    duration: float,
    provider: str = "ollama",
) -> None:
    """Print structured JSON to stdout for scripted/skill consumption."""
    output = {
        "model": model,
        "provider": provider,
        "question": question,
        "answer": answer,
        "search_results": search_results,
        "duration_seconds": round(duration, 2),
    }
    import sys

    print(json.dumps(output, indent=2), file=sys.stdout)


# ---------------------------------------------------------------------------
# Core single-query runner with JSON support
# ---------------------------------------------------------------------------


async def run_local_single_query(
    project_root: Path,
    query: str,
    model: str | None = None,
    timeout: float = 120.0,
    verbose_tools: bool = False,
    json_output: bool = False,
) -> None:
    """Run one query against the local Ollama model and print the answer.

    Args:
        project_root: Project root directory
        query: User question
        model: Ollama model name (auto-detected if None)
        timeout: Request timeout in seconds
        verbose_tools: Show tool call names as they execute
        json_output: Emit machine-readable JSON instead of rich output
    """
    start_time = time.monotonic()

    # ---- Ollama detection ----
    if not await detect_ollama():
        if json_output:
            _print_json_output(
                model=model or "unknown",
                question=query,
                answer="Error: Ollama not running. Start with: ollama serve",
                search_results=[],
                duration=time.monotonic() - start_time,
            )
        else:
            _show_ollama_not_running()
        raise typer.Exit(1)

    detected_model = model or await detect_best_model()
    if not detected_model:
        msg = "No models found. Pull one with: ollama pull gemma3"
        if json_output:
            _print_json_output(
                model="none",
                question=query,
                answer=f"Error: {msg}",
                search_results=[],
                duration=time.monotonic() - start_time,
            )
        else:
            print_error(msg)
        raise typer.Exit(1)

    limits = get_model_limits(detected_model)

    # ---- Load project ----
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. "
            "Run 'mcp-vector-search init' first."
        )
    config = project_manager.load_config()

    # ---- Build LLM client ----
    llm_client = LLMClient(
        provider="ollama",
        model=detected_model,
        timeout=timeout,
    )

    # ---- Build search engine ----
    with suppress_stdout_stderr():
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = create_database(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )
        search_engine = SemanticSearchEngine(
            database=database,
            project_root=project_root,
            similarity_threshold=config.similarity_threshold,
        )

    # ---- Session ----
    session = EnhancedChatSession(LOCAL_SYSTEM_PROMPT)

    if json_output:
        # Run the tool loop and collect results for JSON output
        collected_results: list[dict[str, Any]] = []
        answer = await _process_local_query_json(
            query=query,
            llm_client=llm_client,
            search_engine=search_engine,
            database=database,
            session=session,
            project_root=project_root,
            config=config,
            limits=limits,
            collected_results=collected_results,
        )
        _print_json_output(
            model=detected_model,
            question=query,
            answer=answer,
            search_results=collected_results,
            duration=time.monotonic() - start_time,
        )
    else:
        if not verbose_tools:
            console.print(
                f"\n[dim]Using local {detected_model} for inference (no API key needed)[/dim]\n"
            )
        await _process_local_query(
            query=query,
            llm_client=llm_client,
            search_engine=search_engine,
            database=database,
            session=session,
            project_root=project_root,
            config=config,
            limits=limits,
            verbose_tools=verbose_tools,
        )


# ---------------------------------------------------------------------------
# Interactive REPL for local models
# ---------------------------------------------------------------------------


async def run_local_repl(
    project_root: Path,
    model: str | None = None,
    timeout: float = 120.0,
    verbose_tools: bool = False,
) -> None:
    """Run interactive REPL using local Ollama model.

    Args:
        project_root: Project root directory
        model: Ollama model override
        timeout: Request timeout
        verbose_tools: Show verbose tool output
    """
    # ---- Ollama detection ----
    if not await detect_ollama():
        _show_ollama_not_running()
        raise typer.Exit(1)

    detected_model = model or await detect_best_model()
    if not detected_model:
        print_error("No models found. Pull one with: ollama pull gemma3")
        raise typer.Exit(1)

    limits = get_model_limits(detected_model)

    # ---- Project load ----
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. "
            "Run 'mcp-vector-search init' first."
        )
    config = project_manager.load_config()

    # ---- LLM client ----
    llm_client = LLMClient(
        provider="ollama",
        model=detected_model,
        timeout=timeout,
    )

    # ---- Search engine ----
    with suppress_stdout_stderr():
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = create_database(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )
        search_engine = SemanticSearchEngine(
            database=database,
            project_root=project_root,
            similarity_threshold=config.similarity_threshold,
        )

    session = EnhancedChatSession(LOCAL_SYSTEM_PROMPT)

    _show_intro(detected_model)
    console.print("[dim]Type your questions or /exit to quit[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                command = user_input.lower().split()[0]
                args = user_input[len(command) :].strip()

                if command in ("/exit", "/quit"):
                    console.print("\n[cyan]Goodbye![/cyan]")
                    break
                elif command == "/clear":
                    session.clear()
                    console.print("[green]Conversation cleared.[/green]\n")
                    continue
                elif command == "/task":
                    if args:
                        session.set_task(args)
                        console.print(f"[green]Task set:[/green] {args}\n")
                    else:
                        console.print("[yellow]Usage: /task <description>[/yellow]\n")
                    continue
                elif command == "/status":
                    console.print(
                        f"\n[bold cyan]Model:[/bold cyan] {detected_model} (local)\n"
                        f"[bold cyan]Messages:[/bold cyan] {len(session.messages)}\n"
                    )
                    continue
                else:
                    console.print(f"[yellow]Unknown command: {command}[/yellow]")
                    continue

            await _process_local_query(
                query=user_input,
                llm_client=llm_client,
                search_engine=search_engine,
                database=database,
                session=session,
                project_root=project_root,
                config=config,
                limits=limits,
                verbose_tools=verbose_tools,
            )

        except KeyboardInterrupt:
            console.print("\n\n[cyan]Goodbye![/cyan]")
            break
        except EOFError:
            console.print("\n\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print_error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Compare mode: run same query against local AND cloud provider
# ---------------------------------------------------------------------------


async def run_compare_query(
    project_root: Path,
    query: str,
    local_model: str | None = None,
    cloud_provider: str | None = None,
    cloud_model: str | None = None,
    timeout: float = 120.0,
) -> None:
    """Run query against local Ollama AND cloud provider, show side-by-side.

    Args:
        project_root: Project root directory
        query: User question
        local_model: Override for local model
        cloud_provider: Cloud provider (auto-detected)
        cloud_model: Override for cloud model
        timeout: Timeout in seconds
    """
    from ...core.config_utils import get_openai_api_key, get_openrouter_api_key

    config_dir = project_root / ".mcp-vector-search"
    openai_key = get_openai_api_key(config_dir)
    openrouter_key = get_openrouter_api_key(config_dir)

    # ---- Detect local ----
    ollama_available = await detect_ollama()
    detected_local = local_model or (
        await detect_best_model() if ollama_available else None
    )

    # ---- Detect cloud ----
    cloud_available = False
    effective_cloud_provider = cloud_provider
    if not effective_cloud_provider:
        if is_bedrock_available():
            effective_cloud_provider = "bedrock"
            cloud_available = True
        elif openrouter_key:
            effective_cloud_provider = "openrouter"
            cloud_available = True
        elif openai_key:
            effective_cloud_provider = "openai"
            cloud_available = True
    else:
        cloud_available = True

    if not ollama_available:
        print_warning("Ollama not running — skipping local inference.")
    if not cloud_available:
        print_warning("No cloud provider configured — skipping cloud inference.")

    if not ollama_available and not cloud_available:
        print_error("Neither Ollama nor a cloud provider is available.")
        raise typer.Exit(1)

    # ---- Load project ----
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )
    config = project_manager.load_config()

    with suppress_stdout_stderr():
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = create_database(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )
        search_engine = SemanticSearchEngine(
            database=database,
            project_root=project_root,
            similarity_threshold=config.similarity_threshold,
        )

    console.print(f"\n[bold]Comparing responses for:[/bold] {query}\n")

    async def _run_one(provider: str, mdl: str | None) -> tuple[str, str, float]:
        """Return (provider_label, answer, duration)."""
        t0 = time.monotonic()
        try:
            kwargs: dict[str, Any] = {"provider": provider, "timeout": timeout}
            if mdl:
                kwargs["model"] = mdl
            if provider == "openai":
                kwargs["openai_api_key"] = openai_key
            elif provider == "openrouter":
                kwargs["openrouter_api_key"] = openrouter_key

            client = LLMClient(**kwargs)
            session = EnhancedChatSession(LOCAL_SYSTEM_PROMPT)
            limits = (
                get_model_limits(client.model)
                if provider == "ollama"
                else {"max_iterations": 10, "max_results": 5}
            )

            answer_parts: list[str] = []

            async def _capture_answer() -> None:
                tools = _get_tools()
                session.add_message("user", query)
                messages = session.get_messages()
                max_iter = limits["max_iterations"]

                for _ in range(max_iter):
                    resp = await client.chat_with_tools(messages, tools)
                    choice = resp.get("choices", [{}])[0]
                    msg = choice.get("message", {})
                    tcs = msg.get("tool_calls", [])

                    if tcs:
                        messages.append(msg)
                        for tc in tcs:
                            fn = tc.get("function", {})
                            try:
                                args = json.loads(fn.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                args = {}
                            result = await _execute_tool(
                                fn.get("name", ""),
                                args,
                                search_engine,
                                database,
                                project_root,
                                config,
                                session,
                                client,
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id"),
                                    "content": result,
                                }
                            )
                    else:
                        answer_parts.append(msg.get("content", ""))
                        break

            await _capture_answer()
            answer = "\n".join(answer_parts) or "(no response)"
            return client.model, answer, time.monotonic() - t0
        except Exception as exc:
            return str(mdl or provider), f"Error: {exc}", time.monotonic() - t0

    # Run both concurrently
    tasks = []
    if ollama_available and detected_local:
        tasks.append(_run_one("ollama", detected_local))
    if cloud_available and effective_cloud_provider:
        tasks.append(_run_one(effective_cloud_provider, cloud_model))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Display side by side
    table = Table(show_header=True, header_style="bold cyan", expand=True)

    panels = []
    for r in results:
        if isinstance(r, Exception):
            model_label, answer, dur = "error", str(r), 0.0
        else:
            model_label, answer, dur = r  # type: ignore[misc]
        panels.append(
            Panel(
                Markdown(answer),
                title=f"[bold]{model_label}[/bold] ({dur:.1f}s)",
                border_style="cyan",
            )
        )

    if len(panels) == 2:
        console.print(Columns(panels, equal=True))
    else:
        for p in panels:
            console.print(p)

    _ = table  # Suppress unused-variable warning


# ---------------------------------------------------------------------------
# Internal tool loop helpers
# ---------------------------------------------------------------------------


async def _process_local_query(
    query: str,
    llm_client: LLMClient,
    search_engine: Any,
    database: Any,
    session: EnhancedChatSession,
    project_root: Path,
    config: Any,
    limits: dict[str, int],
    verbose_tools: bool = False,
) -> None:
    """Agentic tool loop for local models (rich output).

    Args:
        query: User question
        llm_client: Configured Ollama LLM client
        search_engine: Vector search engine
        database: Vector DB instance
        session: Chat session
        project_root: Project root
        config: Project config
        limits: max_iterations / max_results per model class
        verbose_tools: Show tool names as they execute
    """
    tools = _get_tools()
    session.add_message("user", query)
    messages = session.get_messages()
    max_iterations = limits.get("max_iterations", 10)

    for _ in range(max_iterations):
        try:
            response = await llm_client.chat_with_tools(messages, tools)
        except Exception as exc:
            print_error(f"Error: {exc}")
            return

        choice = response.get("choices", [{}])[0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                if verbose_tools:
                    args_display = ", ".join(
                        f"{k}={repr(v)[:30]}" for k, v in args.items()
                    )
                    console.print(f"[dim]{fn_name}({args_display})[/dim]")
                else:
                    console.print(".", end="", style="dim")

                result = await _execute_tool(
                    fn_name,
                    args,
                    search_engine,
                    database,
                    project_root,
                    config,
                    session,
                    llm_client,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": result,
                    }
                )
        else:
            final = msg.get("content", "")
            if not final:
                print_error("Empty response from model")
                return

            console.print("\n[bold cyan]Assistant:[/bold cyan]\n")
            with Live(
                "", console=console, auto_refresh=True, vertical_overflow="visible"
            ) as live:
                live.update(Markdown(final))
            console.print()
            session.add_message("assistant", final)
            return

    print_warning("Maximum tool iterations reached.")


async def _process_local_query_json(
    query: str,
    llm_client: LLMClient,
    search_engine: Any,
    database: Any,
    session: EnhancedChatSession,
    project_root: Path,
    config: Any,
    limits: dict[str, int],
    collected_results: list[dict[str, Any]],
) -> str:
    """Agentic tool loop returning the final answer string (for JSON mode).

    Args:
        query: User question
        llm_client: Configured LLM client
        search_engine: Search engine
        database: Vector DB
        session: Chat session
        project_root: Project root
        config: Project config
        limits: Model-specific iteration/result limits
        collected_results: Mutable list to collect search result metadata

    Returns:
        Final answer string from the model
    """
    tools = _get_tools()
    session.add_message("user", query)
    messages = session.get_messages()
    max_iterations = limits.get("max_iterations", 10)

    for _ in range(max_iterations):
        try:
            response = await llm_client.chat_with_tools(messages, tools)
        except Exception as exc:
            return f"Error: {exc}"

        choice = response.get("choices", [{}])[0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                result = await _execute_tool(
                    fn_name,
                    args,
                    search_engine,
                    database,
                    project_root,
                    config,
                    session,
                    llm_client,
                )

                # Capture search result metadata for JSON output
                if fn_name == "search_code":
                    collected_results.append(
                        {
                            "tool": fn_name,
                            "query": args.get("query", ""),
                            "result_preview": result[:300],
                        }
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": result,
                    }
                )
        else:
            return msg.get("content", "(no response)")

    return "(max iterations reached — partial answer may be incomplete)"


# ---------------------------------------------------------------------------
# Typer command entry point
# ---------------------------------------------------------------------------


@chat_local_app.callback(invoke_without_command=True)
def chat_local_main(
    ctx: typer.Context,
    query: str | None = typer.Argument(
        None,
        help="Question to answer (omit to start interactive REPL)",
    ),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory (auto-detected if not specified)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        rich_help_panel="Global Options",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Ollama model to use (default: auto-detect best available)",
        rich_help_panel="Model Options",
    ),
    timeout: float = typer.Option(
        120.0,
        "--timeout",
        help="Request timeout in seconds (local models can be slow)",
        min=10.0,
        max=600.0,
        rich_help_panel="Model Options",
    ),
    verbose_tools: bool = typer.Option(
        False,
        "--verbose-tools",
        "-v",
        help="Show verbose tool call details",
        rich_help_panel="Output Options",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON (for scripted/skill use)",
        rich_help_panel="Output Options",
    ),
    compare: bool = typer.Option(
        False,
        "--compare",
        help="Run query against BOTH local model and configured cloud provider",
        rich_help_panel="Model Options",
    ),
    cloud_provider: str | None = typer.Option(
        None,
        "--cloud-provider",
        help="Cloud provider for --compare mode (auto-detected if not specified)",
        rich_help_panel="Model Options",
    ),
    cloud_model: str | None = typer.Option(
        None,
        "--cloud-model",
        help="Cloud model override for --compare mode",
        rich_help_panel="Model Options",
    ),
) -> None:
    """Chat with your codebase using LOCAL Ollama inference — no API key needed.

    Auto-detects Ollama and selects the best available model (prefers Gemma3).

    [bold cyan]Quick Start:[/bold cyan]
        $ ollama serve               # Start Ollama server
        $ ollama pull gemma3         # Pull a model
        $ mcp-vector-search chat-local

    [bold cyan]Single Query:[/bold cyan]
        $ mcp-vector-search chat-local "where is auth handled?"

    [bold cyan]JSON Output (for scripts):[/bold cyan]
        $ mcp-vector-search chat-local "what does X do?" --json

    [bold cyan]Compare local vs cloud:[/bold cyan]
        $ mcp-vector-search chat-local "query" --compare

    [bold cyan]REPL Commands:[/bold cyan]
        /task <desc>  - Set current task
        /status       - Show model and session info
        /clear        - Clear conversation
        /exit         - Exit
    """
    if ctx.invoked_subcommand is not None:
        return

    # Resolve project root
    if project_root is None:
        if ctx.obj and isinstance(ctx.obj, dict):
            project_root = ctx.obj.get("project_root")
        if project_root is None:
            project_root = Path.cwd()

    try:
        if compare:
            if not query:
                print_error("--compare requires a query argument")
                raise typer.Exit(1)
            asyncio.run(
                run_compare_query(
                    project_root=project_root,
                    query=query,
                    local_model=model,
                    cloud_provider=cloud_provider,
                    cloud_model=cloud_model,
                    timeout=timeout,
                )
            )
        elif query:
            asyncio.run(
                run_local_single_query(
                    project_root=project_root,
                    query=query,
                    model=model,
                    timeout=timeout,
                    verbose_tools=verbose_tools,
                    json_output=json_output,
                )
            )
        else:
            asyncio.run(
                run_local_repl(
                    project_root=project_root,
                    model=model,
                    timeout=timeout,
                    verbose_tools=verbose_tools,
                )
            )

    except (typer.Exit, SystemExit):
        raise
    except Exception as exc:
        logger.error(f"chat-local failed: {exc}")
        print_error(f"chat-local failed: {exc}")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    chat_local_app()
