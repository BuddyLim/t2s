"""Tests for the FastAPI HTTP server wrapping the t2s pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from agent import SqlQuery
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


def _sql_agent(sql: str) -> Runnable[list, SqlQuery]:
    """A fake SQL runnable that always returns the given SQL as a SqlQuery."""
    return RunnableLambda(lambda _messages: SqlQuery(sql=sql))


def _answer_agent(text: str) -> GenericFakeChatModel:
    """A fake answer chat model that always replies with the given text."""
    # GenericFakeChatModel cycles through its message iterator indefinitely.
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


def _ctx(
    source: DataSource, sql_agent: Runnable, answer_agent: BaseChatModel
) -> AppState:
    """Build an AppState with stubbed agents over the fixture source."""
    return AppState(
        settings=Settings(anthropic_api_key="x"),
        source=source,
        schema_text=source.schema_text(),
        dict_text="",
        sql_agent=sql_agent,
        answer_agent=answer_agent,
    )


@pytest.fixture
def client(source: DataSource) -> Iterator[TestClient]:
    """A TestClient with the real lifespan bypassed and a stubbed AppState injected."""
    test_ctx = _ctx(
        source,
        _sql_agent("SELECT name FROM sheet1"),
        _answer_agent("Alice and Bob."),
    )
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
    test_ctx = _ctx(
        source,
        _sql_agent("SELECT still_wrong FROM sheet1"),
        _answer_agent("unused"),
    )
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
