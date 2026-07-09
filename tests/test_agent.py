"""Tests for the text-to-SQL agent's execution-based retry loop."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent import SqlQuery, build_sql_agent, generate_sql
from config import Settings
from data_source import DataSource

# Referencing a fixture by its name as a test argument is the pytest idiom.
# pylint: disable=redefined-outer-name


@pytest.fixture
def source(tmp_path: Path) -> DataSource:
    """Provide a loaded DataSource backed by a small two-row .xlsx fixture."""
    path = tmp_path / "s.xlsx"
    con = duckdb.connect()
    con.execute("INSTALL excel; LOAD excel;")
    con.execute(
        "COPY (SELECT * FROM (VALUES ('Alice', 30), ('Bob', 25)) AS t(name, age)) "
        f"TO '{path}' WITH (FORMAT xlsx, HEADER true)"
    )
    con.close()
    return DataSource(path).load()


def _emit_sql(sql: str, info: AgentInfo) -> ModelResponse:
    """Build a model response that calls the structured-output tool with `sql`."""
    tool = info.output_tools[0]
    return ModelResponse(parts=[ToolCallPart(tool.name, {"sql": sql})])


def test_retries_on_bad_sql_then_succeeds(source: DataSource) -> None:
    """First query references a missing column; the model retries and fixes it."""
    calls: list[str] = []

    def model_fn(_messages, info: AgentInfo) -> ModelResponse:
        # First attempt: invalid column. Second: valid.
        sql = "SELECT missing_col FROM sheet1" if not calls else "SELECT name FROM sheet1"
        calls.append(sql)
        return _emit_sql(sql, info)

    agent = build_sql_agent(Settings(anthropic_api_key="x"))
    with agent.override(model=FunctionModel(model_fn)):
        sql, rows = generate_sql(agent, "list names", "schema", "dict", source)

    assert len(calls) == 2  # retried exactly once after the failure
    assert sql == "SELECT name FROM sheet1"
    assert rows == [{"name": "Alice"}, {"name": "Bob"}]
    source.close()


def test_rejects_write_via_retry(source: DataSource) -> None:
    """A non-SELECT is blocked by the read-only guard and triggers a retry."""
    calls: list[str] = []

    def model_fn(_messages, info: AgentInfo) -> ModelResponse:
        sql = "DROP TABLE sheet1" if not calls else "SELECT name FROM sheet1"
        calls.append(sql)
        return _emit_sql(sql, info)

    agent = build_sql_agent(Settings(anthropic_api_key="x"))
    with agent.override(model=FunctionModel(model_fn)):
        sql, _rows = generate_sql(agent, "list names", "schema", "dict", source)

    assert len(calls) == 2
    assert sql == "SELECT name FROM sheet1"
    source.close()


def test_gives_up_after_max_retries(source: DataSource) -> None:
    """Persistently bad SQL exhausts retries and raises."""

    def model_fn(_messages, info: AgentInfo) -> ModelResponse:
        return _emit_sql("SELECT still_wrong FROM sheet1", info)

    agent = build_sql_agent(Settings(anthropic_api_key="x"))
    with agent.override(model=FunctionModel(model_fn)):
        with pytest.raises(UnexpectedModelBehavior):
            generate_sql(agent, "list names", "schema", "dict", source)
    source.close()


def test_generate_sql_without_dictionary(source: DataSource) -> None:
    """Excel-only mode: an empty dict_text still produces rows without error."""

    def model_fn(_messages, info: AgentInfo) -> ModelResponse:
        return _emit_sql("SELECT name FROM sheet1", info)

    agent = build_sql_agent(Settings(anthropic_api_key="x"))
    with agent.override(model=FunctionModel(model_fn)):
        sql, rows = generate_sql(agent, "list names", "schema", "", source)

    assert sql == "SELECT name FROM sheet1"
    assert rows == [{"name": "Alice"}, {"name": "Bob"}]
    source.close()


def test_output_is_sqlquery_type() -> None:
    """The agent's structured output type is a pydantic model."""
    assert issubclass(SqlQuery, BaseModel)
