# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI for Skillspector — thin wrapper over the LangGraph workflow.

Maps CLI args to initial state, invokes the graph, then maps result to output and exit code.
No business logic; workflow lives in the graph.
"""

from __future__ import annotations

import json
import os
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from langchain_core.runnables import RunnableConfig
from rich.console import Console

from skillspector import __version__
from skillspector.graph import graph
from skillspector.logging_config import get_logger, set_level
from skillspector.nodes.report import (
    _format_json,
    _format_markdown,
    _format_terminal,
)

logger = get_logger(__name__)

app = typer.Typer(
    name="skillspector",
    help="Security scanner for AI agent skills (LangGraph). Detect vulnerabilities before installation.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


class FormatChoice(StrEnum):
    """Output format choices for the CLI."""

    terminal = "terminal"
    json = "json"
    markdown = "markdown"
    sarif = "sarif"


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"SkillSpector v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """
    SkillSpector - Security scanner for AI agent skills (LangGraph).

    Analyze skill bundles to detect vulnerabilities and security risks.
    Supports: Git URL, file URL, .zip file, .md file, or directory.
    """
    pass


def _scan_state(
    input_path: str,
    format: FormatChoice,
    no_llm: bool,
    yara_rules_dir: str | None = None,
) -> dict[str, object]:
    """Build initial graph state from scan CLI args."""
    state: dict[str, object] = {
        "input_path": input_path,
        "output_format": format.value,
        "use_llm": not no_llm,
    }
    if yara_rules_dir is not None:
        state["yara_rules_dir"] = yara_rules_dir
    return state


def _render_format(
    result: dict[str, object],
    fmt: FormatChoice,
) -> str:
    """Render ``result`` in the requested format using the same formatters the
    graph node uses, so multi-format output stays consistent with single-format
    output. Falls back to SARIF JSON when ``report_body`` is absent.
    """
    if result.get("report_body"):
        if fmt == FormatChoice.terminal and not result.get("_format_terminal"):
            pass
        elif fmt == FormatChoice.json and not result.get("_format_json"):
            pass
        elif fmt == FormatChoice.markdown and not result.get("_format_markdown"):
            pass
        else:
            return str(result["report_body"])

    findings = result.get("filtered_findings") or result.get("findings") or []
    component_metadata = result.get("component_metadata") or []
    has_executable_scripts = bool(result.get("has_executable_scripts", False))
    manifest = result.get("manifest") or {}
    skill_path = result.get("skill_path")
    use_llm = bool(result.get("use_llm", True))
    risk_score = result.get("risk_score") or 0
    risk_severity = result.get("risk_severity") or ""
    risk_recommendation = result.get("risk_recommendation") or ""

    if fmt == FormatChoice.terminal:
        return _format_terminal(
            findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
        )
    if fmt == FormatChoice.json:
        return _format_json(
            findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
            use_llm=use_llm,
        )
    if fmt == FormatChoice.markdown:
        return _format_markdown(
            findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
        )
    sarif = result.get("sarif_report")
    return json.dumps(sarif, indent=2) if sarif is not None else ""


def _write_results(
    result: dict[str, object],
    formats: list[FormatChoice],
    outputs: list[Path | None],
) -> None:
    """Write the same scan result in one or more formats.

    Pairing rule: ``formats[i]`` is paired with ``outputs[i]``.  When only one
    output is given it is reused for every format.  When no output is given
    every format is printed to stdout, separated by ``\\n\\n--- <format> ---\\n``.
    """
    if not formats:
        formats = [FormatChoice.terminal]

    if len(outputs) == 1:
        paired_outputs: list[Path | None] = [outputs[0]] * len(formats)
    elif len(outputs) == 0:
        paired_outputs = [None] * len(formats)
    else:
        if len(outputs) != len(formats):
            console.print(
                f"[red]Error:[/red] --output count ({len(outputs)}) must match "
                f"--format count ({len(formats)}) or be exactly 1."
            )
            raise typer.Exit(code=2)
        paired_outputs = outputs

    for fmt, out in zip(formats, paired_outputs, strict=True):
        body = _render_format(result, fmt)
        if out is not None:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(body, encoding="utf-8")
            console.print(f"Report saved to: {out_path} (format={fmt.value})")
        else:
            if fmt == FormatChoice.terminal:
                console.print(body)
            else:
                print(body)


def _cleanup_result(result: dict[str, object]) -> None:
    """Remove temp dir from graph result if set."""
    temp_dir = result.get("temp_dir_for_cleanup")
    if temp_dir and isinstance(temp_dir, str):
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.command()
def scan(
    input_path: Annotated[
        str,
        typer.Argument(
            help="Path or URL to scan. Supports: Git URL, file URL, zip file, .md file, or directory.",
        ),
    ],
    format: Annotated[
        list[FormatChoice] | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format. Repeat to emit multiple formats from a single scan.",
            case_sensitive=False,
        ),
    ] = None,
    output: Annotated[
        list[Path] | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output file path. Repeat to match each --format. "
                "If exactly one --output is given, it is reused for every format. "
                "If none is given, every format is printed to stdout."
            ),
        ),
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option(
            "--no-llm",
            help="Skip LLM analysis (faster, less accurate). Uses static analysis only.",
        ),
    ] = False,
    yara_rules_dir: Annotated[
        Path | None,
        typer.Option(
            "--yara-rules-dir",
            help="Directory containing additional YARA rule files (.yar/.yara) to load alongside built-in rules.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Show detailed progress.",
        ),
    ] = False,
) -> None:
    """
    Scan a skill for security vulnerabilities.

    Examples:

        skillspector scan ./my-skill/
        skillspector scan ./my-skill/ -f json -o report.json
        skillspector scan ./my-skill/ -f json -f markdown -o r.json -o r.md
        skillspector scan ./my-skill/ -f json -f markdown -f terminal
        skillspector scan https://github.com/user/my-skill --no-llm

    Environment variables:

        SKILLSPECTOR_PROVIDER  Active LLM provider: openai | anthropic |
                               nv_build | nv_inference. Defaults to the
                               NVIDIA path (nv_inference, falling back to
                               nv_build in OSS builds).
        SKILLSPECTOR_MODEL     Override the active provider's default
                               model (applies to every analyzer slot).
        SKILLSPECTOR_LOG_LEVEL DEBUG | INFO | WARNING | ERROR (default WARNING).

    Provider credentials (one of):

        OPENAI_API_KEY [+ OPENAI_BASE_URL]   for SKILLSPECTOR_PROVIDER=openai
        ANTHROPIC_API_KEY                    for SKILLSPECTOR_PROVIDER=anthropic
        NVIDIA_INFERENCE_KEY                 for the NVIDIA providers
    """
    result = None
    formats: list[FormatChoice] = format or [FormatChoice.terminal]
    outputs: list[Path | None] = output or []
    try:
        yara_dir = str(yara_rules_dir.resolve()) if yara_rules_dir else None
        primary_format = formats[0]
        state = _scan_state(input_path, primary_format, no_llm, yara_rules_dir=yara_dir)
        if verbose:
            set_level("DEBUG")
            console.print("[dim]Running scan...[/dim]")
        logger.debug(
            "Scan started: input_path=%s, formats=%s, use_llm=%s",
            input_path,
            [f.value for f in formats],
            not no_llm,
        )
        env = os.environ.get("ENV", "dev")
        tags = ["skillspector", f"environment:{env}"]
        extra_tags = os.environ.get("LANGCHAIN_TAGS_EXTRA", "")
        tags.extend(t.strip() for t in extra_tags.split(",") if t.strip())
        trace_config: RunnableConfig = {
            "run_name": "skillspector-scan",
            "tags": tags,
            "metadata": {
                "input_path": input_path,
                "use_llm": not no_llm,
                "output_format": primary_format.value,
                "output_formats": [f.value for f in formats],
                "version": __version__,
            },
        }
        result = graph.invoke(state, config=trace_config)

        _write_results(result, formats, outputs)

        if (result.get("risk_score") or 0) > 50:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    finally:
        if result is not None:
            _cleanup_result(result)


if __name__ == "__main__":
    app()
