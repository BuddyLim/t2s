"""t2s — natural-language querying over an Excel file + Word data dictionary.

Run `python t2s.py` (or `uv run t2s.py`) to start an interactive prompt. Type a
question in plain English; the tool generates SQL, runs it against the Excel data
in DuckDB, and prints a natural-language answer. Pass `--debug` to also see the
generated SQL and the raw result rows.
"""

from __future__ import annotations

import argparse
import sys

from pydantic_ai.exceptions import UnexpectedModelBehavior
from rich.console import Console
from rich.panel import Panel

from agent import build_sql_agent, generate_sql
from answer import answer_question, build_answer_agent
from config import load_settings
from data_dictionary import load_dictionary
from data_source import DataSource

console = Console()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="t2s",
        description="Ask natural-language questions about an Excel workbook.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the generated SQL and raw result rows.",
    )
    return parser.parse_args(argv)


def _handle_question(  # pylint: disable=too-many-arguments  # pipeline context passed as keyword-only args
    question: str,
    *,
    source: DataSource,
    schema_text: str,
    dict_text: str,
    sql_agent,
    answer_agent,
    debug: bool,
) -> None:
    """Run one question through the full pipeline and print the answer."""
    try:
        sql, rows = generate_sql(sql_agent, question, schema_text, dict_text, source)
    except UnexpectedModelBehavior:
        console.print(
            "[red]Could not produce a working query for that question "
            "after several attempts. Try rephrasing it.[/red]"
        )
        return

    if debug:
        console.print(Panel(sql, title="Generated SQL", border_style="cyan"))
        console.print(Panel(str(rows), title="Raw rows", border_style="magenta"))

    answer = answer_question(answer_agent, question, rows)
    console.print(Panel(answer, title="Answer", border_style="green"))


def main(argv: list[str] | None = None) -> int:
    """Parse args, load data, and run the interactive question/answer loop."""
    args = _parse_args(argv)
    settings = load_settings()

    console.print("[bold]Loading data...[/bold]")
    try:
        source = DataSource(settings.excel_path).load()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("Set EXCEL_PATH (and optionally DICT_PATH) in your .env (see .env.example).")
        return 1

    # The data dictionary is optional; run Excel-only when it is absent.
    dict_text = ""
    if settings.dict_path is not None:
        # pylint: disable-next=no-member  # dict_path is Path|None; pylint misreads the pydantic Field type
        if settings.dict_path.exists():
            dict_text = load_dictionary(settings.dict_path)
        else:
            console.print(
                f"[yellow]Data dictionary not found at {settings.dict_path}; "
                "continuing without it.[/yellow]"
            )
    else:
        console.print("[yellow]Data dictionary not set, skipping..[/yellow]")

    schema_text = source.schema_text()
    sql_agent = build_sql_agent(settings)
    answer_agent = build_answer_agent(settings)

    console.print(
        "[green]Ready.[/green] Ask a question, or type "
        "[bold]exit[/bold] / [bold]quit[/bold] to leave.\n"
    )

    try:
        while True:
            try:
                question = console.input("[bold cyan]?[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            try:
                _handle_question(
                    question,
                    source=source,
                    schema_text=schema_text,
                    dict_text=dict_text,
                    sql_agent=sql_agent,
                    answer_agent=answer_agent,
                    debug=args.debug,
                )
            # pylint: disable-next=broad-exception-caught  # keep the REPL alive on any runtime error
            except Exception as exc:  # noqa: BLE001 - surface any runtime error
                console.print(f"[red]Error:[/red] {exc}")
    finally:
        source.close()

    console.print("Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
