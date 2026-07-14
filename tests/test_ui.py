"""Tests for the Gradio UI's pure HTTP helpers (no server, no Gradio launch)."""

from __future__ import annotations

import httpx
import pytest

import ui


def _client(handler) -> httpx.Client:
    """Build an httpx.Client backed by a MockTransport using the given handler."""
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    )


def test_ask_debug_maps_answer_sql_and_rows() -> None:
    """A 200 debug response surfaces the answer, SQL and rows."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answer": "Alice and Bob.",
                "sql": "SELECT name FROM sheet1",
                "rows": [{"name": "Alice"}, {"name": "Bob"}],
            },
        )

    with _client(handler) as client:
        answer, sql, rows = ui.ask("list names", True, client=client)

    assert answer == "Alice and Bob."
    assert sql == "SELECT name FROM sheet1"
    assert rows == [{"name": "Alice"}, {"name": "Bob"}]


def test_ask_without_debug_yields_answer_only() -> None:
    """A 200 non-debug response returns just the answer, no SQL or rows."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "There are 42 rows."})

    with _client(handler) as client:
        answer, sql, rows = ui.ask("how many rows?", False, client=client)

    assert answer == "There are 42 rows."
    assert sql == ""
    assert rows is None


def test_ask_422_maps_to_friendly_detail() -> None:
    """A 422 response surfaces the server's `detail` as the answer."""
    detail = "Could not produce a working query for that question. Try rephrasing it."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": detail})

    with _client(handler) as client:
        answer, sql, rows = ui.ask("gibberish", True, client=client)

    assert answer == detail
    assert sql == ""
    assert rows is None


def test_ask_connection_error_maps_to_cant_reach(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport error becomes the friendly 'Can't reach' message."""
    monkeypatch.setenv("T2S_API_URL", "http://localhost:9999")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _client(handler) as client:
        answer, sql, rows = ui.ask("list names", False, client=client)

    assert answer == "Can't reach the t2s server at http://localhost:9999. Is it running?"
    assert sql == ""
    assert rows is None


def test_ask_empty_question_short_circuits_without_http() -> None:
    """A blank question returns a gentle prompt and never touches the network."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"answer": "unused"})

    with _client(handler) as client:
        answer, sql, rows = ui.ask("   ", True, client=client)

    assert answer == "Please enter a question."
    assert sql == ""
    assert rows is None
    assert calls == 0
