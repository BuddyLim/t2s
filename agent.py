"""Text-to-SQL agent: turn a natural-language question into DuckDB SQL.

Uses the fast/cheap model. Output is validated as a Pydantic model so we always
get a single SQL string field back, never free-form prose.

The output validator *executes* the query against DuckDB. If the SQL is invalid
(wrong column, bad function, syntax error) or not read-only, it raises ModelRetry
with the actual error, so PydanticAI re-prompts the model to fix it — up to
`_MAX_RETRIES` times. On success the result rows are captured on the deps object,
so the query only runs once for the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import duckdb
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

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


@dataclass
class SqlDeps:
    """Dependencies for the SQL agent; also captures the executed rows."""

    source: DataSource
    rows: list[dict] = field(default_factory=list)


def build_sql_agent(settings: Settings) -> Agent[SqlDeps, SqlQuery]:
    """Create the text-to-SQL agent, with execution-based retry against DuckDB."""
    # `ty` cannot infer the generic `OutputDataT` from `output_type=SqlQuery`
    # (the `OutputSpec` alias is recursive), so it falls back to the default
    # `Agent[SqlDeps, str]`. Cast to pin the real output type; no runtime effect.
    agent = cast(
        "Agent[SqlDeps, SqlQuery]",
        Agent(
            build_model(settings, settings.sql_model),
            output_type=SqlQuery,
            deps_type=SqlDeps,
            instructions=_INSTRUCTIONS,
            retries=_MAX_RETRIES,
        ),
    )

    @agent.output_validator
    def _execute_and_validate(ctx: RunContext[SqlDeps], output: SqlQuery) -> SqlQuery:
        """Run the query; on any failure, ask the model to fix it."""
        try:
            ctx.deps.rows = ctx.deps.source.run_sql(output.sql)
        except UnsafeQueryError as exc:
            raise ModelRetry(
                f"The query was rejected because it is not read-only: {exc}. "
                "Return a single read-only SELECT statement."
            ) from exc
        except duckdb.Error as exc:
            raise ModelRetry(
                f"The query failed to run in DuckDB with error: {exc}. "
                "Fix the SQL, using only the tables and columns from the schema."
            ) from exc
        return output

    return agent


def generate_sql(
    agent: Agent[SqlDeps, SqlQuery],
    question: str,
    schema_text: str,
    dict_text: str,
    source: DataSource,
) -> tuple[str, list[dict]]:
    """Generate a working SQL query and return it with its result rows.

    The agent retries (up to _MAX_RETRIES) if DuckDB rejects the SQL. Raises
    pydantic_ai.exceptions.UnexpectedModelBehavior if all retries are exhausted.
    """
    deps = SqlDeps(source=source)
    sections = [f"# Database schema\n{schema_text}"]
    if dict_text.strip():
        sections.append(f"# Data dictionary\n{dict_text}")
    sections.append(f"# Question\n{question}")
    prompt = "\n\n".join(sections)
    result = agent.run_sync(prompt, deps=deps)
    return result.output.sql.strip(), deps.rows
