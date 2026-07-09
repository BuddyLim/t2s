"""Tests for the FastAPI HTTP server wrapping the t2s pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent import build_sql_agent
from answer import build_answer_agent
from app import AppState, app, get_ctx
from config import Settings
from data_source import DataSource

# Referencing a fixture by its name as a test argument is the pytest idiom.
# pylint: disable=redefined-outer-name


@pytest.fixture
def source(tmp_path: Path) -> Iterator[DataSource]:
    """Provide a loaded DataSource backed by a small two-row .xlsx fixture."""
    path = tmp_path / "s.xlsx"
    con = duckdb.connect()
    con.execute("INSTALL excel; LOAD excel;")
    con.execute(
        "COPY (SELECT * FROM (VALUES ('Alice', 30), ('Bob', 25)) AS t(name, age)) "
        f"TO '{path}' WITH (FORMAT xlsx, HEADER true)"
    )
    con.close()
    src = DataSource(path).load()
    yield src
    src.close()


def _emit_sql(sql: str, info: AgentInfo) -> ModelResponse:
    """Build a model response that calls the structured-output tool with `sql`."""
    tool = info.output_tools[0]
    return ModelResponse(parts=[ToolCallPart(tool.name, {"sql": sql})])


def _emit_text(text: str) -> ModelResponse:
    """Build a plain-text model response for the answer agent."""
    return ModelResponse(parts=[TextPart(content=text)])


@pytest.fixture
def client(source: DataSource) -> Iterator[TestClient]:
    """A TestClient with the real lifespan bypassed and a stubbed AppState injected."""
    settings = Settings(anthropic_api_key="x")
    sql_agent = build_sql_agent(settings)
    answer_agent = build_answer_agent(settings)

    def sql_fn(_messages, info: AgentInfo) -> ModelResponse:
        return _emit_sql("SELECT name FROM sheet1", info)

    def answer_fn(_messages, _info: AgentInfo) -> ModelResponse:
        return _emit_text("Alice and Bob.")

    test_ctx = AppState(
        settings=settings,
        source=source,
        schema_text=source.schema_text(),
        dict_text="",
        sql_agent=sql_agent,
        answer_agent=answer_agent,
    )

    with ExitStack() as stack:
        stack.enter_context(sql_agent.override(model=FunctionModel(sql_fn)))
        stack.enter_context(answer_agent.override(model=FunctionModel(answer_fn)))
        app.dependency_overrides[get_ctx] = lambda: test_ctx
        try:
            yield TestClient(app)
        finally:
            del app.dependency_overrides[get_ctx]


def test_health(client: TestClient) -> None:
    """The health endpoint reports ok status and at least one loaded table."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["tables"] >= 1


def test_query_without_debug_omits_sql_and_rows(client: TestClient) -> None:
    """Without debug, only the answer is returned; sql/rows keys are absent."""
    resp = client.post("/query", json={"question": "list names"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body and body["answer"]
    assert "sql" not in body
    assert "rows" not in body


def test_query_with_debug_includes_sql_and_rows(client: TestClient) -> None:
    """With debug=True, the generated SQL and raw rows are also returned."""
    resp = client.post("/query", json={"question": "list names", "debug": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["sql"] == "SELECT name FROM sheet1"
    assert body["rows"] == [{"name": "Alice"}, {"name": "Bob"}]


def test_query_gives_up_after_max_retries_returns_422(source: DataSource) -> None:
    """Persistently invalid SQL exhausts retries and surfaces as a 422."""
    settings = Settings(anthropic_api_key="x")
    sql_agent = build_sql_agent(settings)
    answer_agent = build_answer_agent(settings)

    def bad_sql_fn(_messages, info: AgentInfo) -> ModelResponse:
        return _emit_sql("SELECT still_wrong FROM sheet1", info)

    test_ctx = AppState(
        settings=settings,
        source=source,
        schema_text=source.schema_text(),
        dict_text="",
        sql_agent=sql_agent,
        answer_agent=answer_agent,
    )

    with ExitStack() as stack:
        stack.enter_context(sql_agent.override(model=FunctionModel(bad_sql_fn)))
        app.dependency_overrides[get_ctx] = lambda: test_ctx
        try:
            resp = TestClient(app).post("/query", json={"question": "list names"})
        finally:
            del app.dependency_overrides[get_ctx]

    assert resp.status_code == 422
    assert isinstance(resp.json()["detail"], str)


def test_query_rejects_empty_question(client: TestClient) -> None:
    """An empty question fails pydantic's min_length validation."""
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422
