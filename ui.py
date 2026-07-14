"""Streamlit web UI wrapping the t2s FastAPI server.

A thin HTTP client: it POSTs questions to a running t2s server and renders the
answer (optionally with the generated SQL and rows). It never imports the
pipeline — it only knows the HTTP contract exposed by `app.py`.

Run the API first (`uv run fastapi run`), then in another terminal
`uv run streamlit run ui.py`, and open http://localhost:8501.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"

# What `ask()` returns and the UI renders: (answer_markdown, sql, rows).
AskResult = tuple[str, str, list[dict] | None]


def api_url() -> str:
    """Return the base URL of the t2s API from `T2S_API_URL` (or the default)."""
    return os.environ.get("T2S_API_URL", DEFAULT_API_URL)


def friendly_error(exc: Exception, url: str) -> str:
    """Map an httpx transport exception to a friendly, user-facing message."""
    if isinstance(exc, httpx.ConnectError | httpx.TimeoutException | httpx.RequestError):
        return f"Can't reach the t2s server at {url}. Is it running?"
    return "Something went wrong talking to the t2s server."


def parse_response(resp: httpx.Response, show_details: bool) -> AskResult:
    """Map a `/query` HTTP response to an `(answer, sql, rows)` tuple.

    Only populates sql/rows when `show_details` is on and the server included
    them (the server omits them unless debug was requested).
    """
    if resp.status_code == 200:
        body = resp.json()
        answer = body.get("answer", "")
        if show_details:
            return answer, body.get("sql") or "", body.get("rows")
        return answer, "", None
    if resp.status_code == 422:
        return _detail(resp, "Try rephrasing your question."), "", None
    if resp.status_code == 500:
        return "The server hit an unexpected error.", "", None
    fallback = f"The server returned an unexpected status ({resp.status_code})."
    return _detail(resp, fallback), "", None


def _detail(resp: httpx.Response, fallback: str) -> str:
    """Return the `detail` field of an error response, or a fallback string."""
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    return detail if isinstance(detail, str) and detail else fallback


def ask(
    question: str,
    show_details: bool,
    client: httpx.Client | None = None,
) -> AskResult:
    """Ask the t2s server a question and return `(answer, sql, rows)`.

    Handles every error path gracefully — an empty question short-circuits
    without an HTTP call, and transport/HTTP errors become friendly strings.
    A `client` may be injected (e.g. for tests); otherwise one is created.
    """
    if not question or not question.strip():
        return "Please enter a question.", "", None

    url = api_url()
    owns_client = client is None
    client = client or httpx.Client(base_url=url, timeout=120.0)
    try:
        resp = client.post(
            "/query",
            json={"question": question, "debug": show_details},
        )
        return parse_response(resp, show_details)
    except httpx.HTTPError as exc:
        return friendly_error(exc, url), "", None
    finally:
        if owns_client:
            client.close()


def check_health(client: httpx.Client | None = None) -> str:
    """Return a short status line describing the server's health."""
    url = api_url()
    owns_client = client is None
    client = client or httpx.Client(base_url=url, timeout=10.0)
    try:
        resp = client.get("/health")
        if resp.status_code == 200:
            tables = resp.json().get("tables", 0)
            return f"Connected — {tables} tables loaded"
        return f"Server responded with status {resp.status_code}."
    except httpx.HTTPError as exc:
        return friendly_error(exc, url)
    finally:
        if owns_client:
            client.close()


def render() -> None:
    """Render the Streamlit UI, wired to the pure helpers above."""
    st.set_page_config(page_title="t2s", page_icon="📊")
    st.title("t2s — ask your spreadsheet")
    st.caption("Ask plain-English questions about the loaded Excel workbook.")
    st.write(check_health())

    with st.form("ask"):
        question = st.text_area(
            "Question",
            placeholder="e.g. which city has the most customers?",
        )
        show_details = st.checkbox("Show SQL & rows", value=False)
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        with st.spinner("Thinking…"):
            answer, sql, rows = ask(question, show_details)
        st.markdown(answer)
        if show_details:
            if sql:
                st.code(sql, language="sql")
            if rows is not None:
                st.dataframe(rows)


if __name__ == "__main__":
    render()
