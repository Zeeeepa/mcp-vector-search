#!/usr/bin/env python3
"""Benchmark mcp-vector-search vs ripgrep across projects.

Measures:
  - speed (wall-clock ms): cold (first call, includes model load) and
    warm (subsequent calls after OS has cached the model)
  - context pollution (% irrelevant + token estimate)

Usage:
  uv run python scripts/benchmark_search_vs_rg.py
  uv run python scripts/benchmark_search_vs_rg.py --project mvs
  uv run python scripts/benchmark_search_vs_rg.py --project cto
  uv run python scripts/benchmark_search_vs_rg.py --project duetto
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Resolve ripgrep binary (some shells alias `rg` as a function not visible to subprocess)
RG_BIN = shutil.which("rg") or next(
    (
        p
        for p in [
            "/opt/homebrew/bin/rg",
            "/usr/local/bin/rg",
            "/Users/masa/.local/share/cursor-agent/versions/2026.03.30-a5d3e17/rg",
        ]
        if os.path.exists(p)
    ),
    "rg",
)

# ---- Configuration -----------------------------------------------------------

_ALL_PROJECTS: list[tuple[str, str, str, str]] = [
    # (short_key, label, path, mvs_invocation)
    (
        "mvs",
        "mcp-vector-search (small, code-only)",
        "/Users/masa/Projects/mcp-vector-search",
        "uv-run",
    ),
    (
        "cto",
        "Duetto/cto (mixed: code + docs)",
        str(Path.home() / "Duetto/cto"),
        "direct",
    ),
    (
        "duetto",
        "Duetto/repos/duetto (large codebase)",
        str(Path.home() / "Duetto/repos/duetto"),
        "direct",
    ),
]

# Keys for the --project flag
PROJECT_KEYS: dict[str, tuple[str, str, str]] = {  # (label, path, invocation)
    row[0]: (row[1], row[2], row[3]) for row in _ALL_PROJECTS
}

WARMUP_QUERY = "error handling"

QUERIES: list[str] = [
    "authentication and authorization logic",
    "database connection handling",
    "error handling and retry logic",
    "configuration loading and environment variables",
    "test fixtures and mock setup",
]

QUERY_KEYWORDS: dict[str, str] = {
    "authentication and authorization logic": "auth",
    "database connection handling": "database",
    "error handling and retry logic": "retry",
    "configuration loading and environment variables": "config",
    "test fixtures and mock setup": "fixture",
}

MVS_LIMIT = 10
RG_HEAD = 20
POLLUTION_THRESHOLD = 0.5
TOP_FILES_KEEP = 5

# ---- Data --------------------------------------------------------------------


@dataclass
class RunResult:
    label: str
    speed_ms: float
    result_count: int
    token_estimate: int
    pollution_pct: float
    note: str = ""


def estimate_tokens_from_chars(chars: int) -> int:
    return max(0, chars // 4)


# ---- Runners -----------------------------------------------------------------


def _invoke_mvs(
    project_path: str, invocation: str, query: str
) -> tuple[float, str, int]:
    """Run one MVS subprocess query.

    Returns (elapsed_ms, stdout, returncode).
    """
    # Use --search-mode vector to get raw cosine similarity scores.
    # Hybrid mode applies RRF normalization which compresses scores toward
    # the middle regardless of relevance, breaking the < 0.5 pollution
    # threshold (every result looks "polluted" even when relevant).
    if invocation == "uv-run":
        cmd = [
            "uv",
            "run",
            "mcp-vector-search",
            "search",
            "--limit",
            str(MVS_LIMIT),
            "--search-mode",
            "vector",
            "--json",
            query,
        ]
        cwd = project_path
    else:
        cmd = [
            "mcp-vector-search",
            "search",
            "-p",
            project_path,
            "--limit",
            str(MVS_LIMIT),
            "--search-mode",
            "vector",
            "--json",
            query,
        ]
        cwd = None

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return (120_000.0, "", -1)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return (elapsed_ms, proc.stdout or "", proc.returncode)


def _parse_mvs_output(raw: str, elapsed_ms: float, label: str) -> RunResult:
    """Parse JSON output from an MVS subprocess call into a RunResult."""
    bracket = raw.find("[")
    if bracket < 0:
        return RunResult(label, elapsed_ms, 0, 0, 100.0, note="no-json")

    try:
        data = json.loads(raw[bracket:])
    except json.JSONDecodeError:
        end = raw.rfind("]")
        try:
            data = json.loads(raw[bracket : end + 1])
        except Exception:
            return RunResult(label, elapsed_ms, 0, 0, 100.0, note="parse-fail")

    if not isinstance(data, list):
        return RunResult(label, elapsed_ms, 0, 0, 100.0, note="not-list")

    n = len(data)
    if n == 0:
        return RunResult(label, elapsed_ms, 0, 0, 0.0, note="empty")

    total_chars = 0
    low_score = 0
    for item in data:
        content = item.get("content", "") or ""
        ctx_b = " ".join(item.get("context_before") or [])
        ctx_a = " ".join(item.get("context_after") or [])
        total_chars += len(content) + len(ctx_b) + len(ctx_a)
        score = float(item.get("similarity_score", 0.0))
        if score < POLLUTION_THRESHOLD:
            low_score += 1

    pollution = (low_score / n) * 100.0
    return RunResult(
        label, elapsed_ms, n, estimate_tokens_from_chars(total_chars), pollution
    )


def run_mvs_cold_warm(
    project_path: str,
    invocation: str,
    query: str,
    warmup_done: bool,
) -> tuple[RunResult | None, RunResult]:
    """Run MVS for one query.

    If this is the first call for this project (warmup_done=False), runs a
    warm-up query first (WARMUP_QUERY) to load the model into OS cache, then
    runs the real query and returns both a cold result (the warm-up) and the
    warm result.

    On subsequent calls (warmup_done=True), returns (None, warm_result) because
    the cold cost was already captured.
    """
    if not warmup_done:
        # Cold: run warm-up query and capture that time as the "cold" cost
        cold_ms, cold_raw, cold_rc = _invoke_mvs(project_path, invocation, WARMUP_QUERY)
        if cold_rc == -1:
            cold_result: RunResult | None = RunResult(
                "MVS-cold", cold_ms, 0, 0, 100.0, note="TIMEOUT"
            )
        else:
            cold_result = _parse_mvs_output(cold_raw, cold_ms, "MVS-cold")

        # Warm: immediately run the real query (model is now in OS cache)
        warm_ms, warm_raw, warm_rc = _invoke_mvs(project_path, invocation, query)
        if warm_rc == -1:
            warm_result = RunResult("MVS-warm", warm_ms, 0, 0, 100.0, note="TIMEOUT")
        else:
            warm_result = _parse_mvs_output(warm_raw, warm_ms, "MVS-warm")

        return cold_result, warm_result
    else:
        # Warm-only: model already cached from warm-up
        warm_ms, warm_raw, warm_rc = _invoke_mvs(project_path, invocation, query)
        if warm_rc == -1:
            warm_result = RunResult("MVS-warm", warm_ms, 0, 0, 100.0, note="TIMEOUT")
        else:
            warm_result = _parse_mvs_output(warm_raw, warm_ms, "MVS-warm")

        return None, warm_result


def run_ripgrep(project_path: str, query: str) -> RunResult:
    keyword = QUERY_KEYWORDS.get(query, query.split()[0])
    cmd = [
        RG_BIN,
        "-l",
        "-i",
        keyword,
        project_path,
        "--type",
        "py",
        "--type",
        "java",
        "--type",
        "ts",
        "--type",
        "js",
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return RunResult("RG", 60_000.0, 0, 0, 100.0, note="TIMEOUT")
    elapsed_ms = (time.perf_counter() - start) * 1000

    files = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    files = files[:RG_HEAD]
    n = len(files)
    if n == 0:
        return RunResult("RG", elapsed_ms, 0, 0, 0.0, note="empty")

    total_chars = 0
    for f in files:
        try:
            res = subprocess.run(
                [RG_BIN, "-n", "-C", "1", "-i", keyword, f],
                capture_output=True,
                text=True,
                timeout=10,
            )
            total_chars += len(res.stdout or "")
        except subprocess.TimeoutExpired:
            continue

    pollution = (max(0, n - TOP_FILES_KEEP) / n) * 100.0
    return RunResult(
        "RG", elapsed_ms, n, estimate_tokens_from_chars(total_chars), pollution
    )


# ---- Reporting ---------------------------------------------------------------

WIDTH = 86


def hline(ch: str = "═") -> str:
    return "╠" + ch * WIDTH + "╣"


def boxline(text: str, edge: str = "║") -> str:
    return edge + " " + text.ljust(WIDTH - 1) + edge


def fmt_row(
    label: str, cold_ms: float | None, warm_ms: float, rg_ms: float, r: RunResult
) -> str:
    """Format a single benchmark row with cold/warm/rg columns."""
    cold_str = f"{cold_ms:>7.0f}" if cold_ms is not None else "      -"
    return (
        f"  {label:<8} "
        f"cold={cold_str}ms  "
        f"warm={warm_ms:>7.0f}ms  "
        f"rg={rg_ms:>7.0f}ms  | "
        f"{r.result_count:>3}res ~{r.token_estimate:>6}tok "
        f"{r.pollution_pct:>5.1f}%poll"
    )


def winner_warm(warm: RunResult, rg: RunResult) -> str:
    """Determine winner based on warm MVS times (fair comparison)."""
    if warm.note == "TIMEOUT" or warm.result_count == 0:
        return "RipGrep (MVS unavailable)"
    if rg.note == "TIMEOUT" or rg.result_count == 0:
        return "MVS (RG unavailable)"
    speed_w = "MVS" if warm.speed_ms < rg.speed_ms else "RipGrep"
    qual_w = "MVS" if warm.pollution_pct < rg.pollution_pct else "RipGrep"
    if speed_w == qual_w:
        return f"{speed_w} (speed+quality)"
    return f"{qual_w} (quality), {speed_w} (speed)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark mcp-vector-search vs ripgrep"
    )
    parser.add_argument(
        "--project",
        choices=list(PROJECT_KEYS.keys()),
        default=None,
        help="Run on a single project (mvs | cto | duetto). Omit to run all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.project:
        label, path, invocation = PROJECT_KEYS[args.project]
        projects: list[tuple[str, str, str]] = [(label, path, invocation)]
    else:
        projects = [(row[1], row[2], row[3]) for row in _ALL_PROJECTS]

    print("╔" + "═" * WIDTH + "╗")
    print(boxline("SEARCH BENCHMARK: mcp-vector-search vs ripgrep"))
    print(boxline("MVS cold = first subprocess call (model load included)"))
    print(boxline("MVS warm = subsequent calls (model in OS cache)"))

    all_cold: list[RunResult] = []
    all_warm: list[RunResult] = []
    all_rg: list[RunResult] = []

    for label, path, invocation in projects:
        print(hline())
        print(boxline(f"Project: {label}"))
        print(boxline(f"Path:    {path}"))
        print(hline())

        if not Path(path).exists():
            print(boxline("SKIPPED: path does not exist"))
            continue

        cold_result_for_project: RunResult | None = None
        warmup_done = False

        for query in QUERIES:
            print(boxline(f'Query: "{query}"', edge="│"))

            cold, warm = run_mvs_cold_warm(
                project_path=path,
                invocation=invocation,
                query=query,
                warmup_done=warmup_done,
            )
            rg = run_ripgrep(path, query)
            all_warm.append(warm)
            all_rg.append(rg)

            if not warmup_done:
                # cold is the warm-up run result; store it once per project
                cold_result_for_project = cold
                warmup_done = True
                if cold is not None:
                    all_cold.append(cold)
                    # Show startup line above the first real query results
                    print(
                        f"  [startup/cold] warm-up query '{WARMUP_QUERY}': "
                        f"{cold.speed_ms:.0f}ms (model load)"
                    )

            # Show cold_ms only on the first query row (the warm-up call)
            display_cold = (
                cold_result_for_project.speed_ms
                if cold is not None and cold_result_for_project is not None
                else None
            )

            print(fmt_row("MVS:", display_cold, warm.speed_ms, rg.speed_ms, warm))
            print(f"  Winner (warm): {winner_warm(warm, rg)}")
            print("│" + "-" * WIDTH + "│")

    # Summary
    def avg(rs: list[RunResult], attr: str) -> float:
        vals = [
            getattr(r, attr)
            for r in rs
            if r.note not in ("TIMEOUT",) and r.result_count > 0
        ]
        return sum(vals) / len(vals) if vals else 0.0

    cold_speed = avg(all_cold, "speed_ms")
    warm_speed = avg(all_warm, "speed_ms")
    rg_speed = avg(all_rg, "speed_ms")
    warm_tokens = avg(all_warm, "token_estimate")
    rg_tokens = avg(all_rg, "token_estimate")
    warm_poll = avg(all_warm, "pollution_pct")
    rg_poll = avg(all_rg, "pollution_pct")

    print(hline())
    print(boxline("SUMMARY (averages across selected projects × queries)"))
    print("│" + "-" * WIDTH + "│")

    def speed_phrase(a_ms: float, b_ms: float, a_label: str, b_label: str) -> str:
        if not a_ms or not b_ms:
            return "n/a"
        if a_ms > b_ms:
            return f"{b_label} {a_ms / b_ms:.1f}x faster"
        return f"{a_label} {b_ms / a_ms:.1f}x faster"

    tok_phrase = "n/a"
    if rg_tokens and warm_tokens:
        if rg_tokens > warm_tokens:
            tok_phrase = f"MVS {rg_tokens / warm_tokens:.1f}x less noise"
        else:
            tok_phrase = f"RG {warm_tokens / rg_tokens:.1f}x less noise"

    print(
        boxline(
            f"avg speed (cold):  MVS {cold_speed:>7.0f}ms  vs  RG {rg_speed:>7.0f}ms"
            f"  ({speed_phrase(cold_speed, rg_speed, 'MVS', 'RG')})",
            edge="│",
        )
    )
    print(
        boxline(
            f"avg speed (warm):  MVS {warm_speed:>7.0f}ms  vs  RG {rg_speed:>7.0f}ms"
            f"  ({speed_phrase(warm_speed, rg_speed, 'MVS', 'RG')})",
            edge="│",
        )
    )
    print(
        boxline(
            f"avg tokens:        MVS {warm_tokens:>7.0f}     vs  RG {rg_tokens:>7.0f}"
            f"     ({tok_phrase})",
            edge="│",
        )
    )
    print(
        boxline(
            f"avg pollution:     MVS {warm_poll:>5.1f}%        vs  RG {rg_poll:>5.1f}%",
            edge="│",
        )
    )
    print("╚" + "═" * WIDTH + "╝")


if __name__ == "__main__":
    main()
