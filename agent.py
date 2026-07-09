"""Text-to-SQL agent: turn a natural-language question into DuckDB SQL.

Uses the fast/cheap model. Output is validated as a Pydantic model so we always
get a single SQL string field back, never free-form prose.

`generate_sql` runs an execute-and-retry loop: it asks the model for a `SqlQuery`,
*executes* it against DuckDB, and if the SQL is invalid (wrong column, bad
function, syntax error) or not read-only, it feeds the actual error back into the
conversation and re-prompts — up to `_MAX_RETRIES` times (so `_MAX_RETRIES + 1`
attempts total). On success the result rows are returned alongside the SQL, so the
query only runs once for the whole pipeline. If every attempt fails, it raises
`SqlGenerationError`.
"""

from __future__ import annotations

from typing import cast

import duckdb
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from config import Settings, build_model
from data_source import DataSource, UnsafeQueryError

_MAX_RETRIES = 3

_INSTRUCTIONS = """\
You are an expert data analyst who writes DuckDB SQL.

You are given:
  1. The database schema (tables and columns) derived from an Excel workbook.
  2. Optionally, a data dictionary explaining what the columns mean. It may be
     absent, in which case rely on the schema alone.

Given a user's question, produce ONE read-only DuckDB SQL query that answers it.

Rules:
- Output a SELECT (or WITH ... SELECT) statement only. Never write INSERT, UPDATE,
  DELETE, CREATE, DROP, or any statement that modifies data.
- Use exactly the table and column names from the provided schema.
- Prefer explicit column lists and add sensible LIMITs for open-ended questions.
- When a data dictionary is provided, use it to map business terms in the
  question to real columns.
- Return only the SQL in the `sql` field; no explanation.
- If told a previous query failed, read the error carefully and correct the SQL.
"""


class SqlQuery(BaseModel):
    """A validated, read-only SQL query answering the user's question."""

    sql: str = Field(description="A single read-only DuckDB SELECT statement.")


class SqlGenerationError(RuntimeError):
    """Raised when the model can't produce a working query within the retry budget."""


class _RetrySql(Exception):
    """Internal signal that a generated query failed; carries the model feedback.

    `feedback` is the message we send back to the model so it can fix the SQL.
    """

    def __init__(self, feedback: str) -> None:
        super().__init__(feedback)
        self.feedback = feedback


def _execute(source: DataSource, sql: str) -> list[dict]:
    """Run `sql` read-only and return the rows.

    Translates a rejection or DuckDB error into a `_RetrySql` signal (chained to
    the original exception) so the caller can re-prompt the model instead of
    handling two exception types inline.
    """
    try:
        return source.run_sql(sql)
    except UnsafeQueryError as exc:
        raise _RetrySql(
            f"The query was rejected because it is not read-only: {exc}. "
            "Return a single read-only SELECT statement."
        ) from exc
    except duckdb.Error as exc:
        raise _RetrySql(
            f"The query failed to run in DuckDB with error: {exc}. "
            "Fix the SQL, using only the tables and columns from the schema."
        ) from exc


def build_sql_agent(settings: Settings) -> Runnable[list, SqlQuery]:
    """Create the text-to-SQL runnable: a chat model bound to the SqlQuery schema.

    `.invoke(messages)` returns a `SqlQuery`. The execute-and-retry loop lives in
    `generate_sql`, which drives this runnable.
    """
    model: BaseChatModel = build_model(settings, settings.sql_model)
    # with_structured_output is typed as returning `dict | BaseModel`; pin the
    # concrete SqlQuery output for callers. No runtime effect.
    return cast(
        "Runnable[list, SqlQuery]",
        model.with_structured_output(SqlQuery),
    )


def _build_prompt(question: str, schema_text: str, dict_text: str) -> str:
    """Assemble the schema, optional dictionary, and question into one prompt."""
    sections = [f"# Database schema\n{schema_text}"]
    if dict_text.strip():
        sections.append(f"# Data dictionary\n{dict_text}")
    sections.append(f"# Question\n{question}")
    return "\n\n".join(sections)


def generate_sql(
    agent: Runnable[list, SqlQuery],
    question: str,
    schema_text: str,
    dict_text: str,
    source: DataSource,
) -> tuple[str, list[dict]]:
    """Generate a working SQL query and return it with its result rows.

    Retries (up to _MAX_RETRIES) if DuckDB rejects the SQL, feeding the error back
    into the conversation each time. Raises SqlGenerationError if all attempts fail.
    """
    messages: list = [
        SystemMessage(_INSTRUCTIONS),
        HumanMessage(_build_prompt(question, schema_text, dict_text)),
    ]
    last_exc: BaseException | None = None

    for _ in range(_MAX_RETRIES + 1):
        sql = agent.invoke(messages).sql.strip()
        try:
            rows = _execute(source, sql)
        except _RetrySql as retry:
            # Feed the failure back as clean conversation turns and try again. We
            # append a plain AIMessage (the SQL text), not the raw tool-call
            # message, to avoid a dangling tool_use without a matching tool_result.
            last_exc = retry.__cause__ or retry
            messages.append(AIMessage(content=f"Previous attempt:\n{sql}"))
            messages.append(HumanMessage(content=retry.feedback))
            continue
        return sql, rows

    raise SqlGenerationError(
        f"Could not produce a working query after {_MAX_RETRIES + 1} attempts."
    ) from last_exc
