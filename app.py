"""FastAPI HTTP server exposing the t2s natural-language-to-SQL pipeline.

Loads the Excel workbook (and optional Word data dictionary) once at startup,
builds the SQL and answer agents, and serves `/health` and `/query` over HTTP.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from agent import SqlGenerationError, SqlQuery, build_sql_agent, generate_sql
from answer import answer_question, build_answer_agent
from config import Settings, load_settings
from data_dictionary import load_dictionary
from data_source import DataSource

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """A natural-language question to answer against the loaded workbook."""

    question: str = Field(min_length=1)
    debug: bool = False


class QueryResponse(BaseModel):
    """The answer to a question, optionally with the generated SQL and rows."""

    answer: str
    sql: str | None = None
    rows: list[dict] | None = None


class HealthResponse(BaseModel):
    """Server health and basic dataset info."""

    status: str
    tables: int


@dataclass
class AppState:
    """Everything the pipeline needs, built once at startup."""

    settings: Settings
    source: DataSource
    schema_text: str
    dict_text: str
    sql_agent: Runnable[list, SqlQuery]
    answer_agent: BaseChatModel


def _load_dict_text(settings: Settings) -> str:
    """Resolve the data dictionary text, matching the previous CLI's behavior."""
    if settings.dict_path is None:
        return ""
    if not settings.dict_path.exists():
        logger.warning(
            "Data dictionary not found at %s; continuing without it.",
            settings.dict_path,
        )
        return ""
    return load_dictionary(settings.dict_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load data and build agents once at startup; close the connection on shutdown."""
    settings = load_settings()
    source = DataSource(settings.excel_path).load()
    try:
        schema_text = source.schema_text()
        dict_text = _load_dict_text(settings)
        sql_agent = build_sql_agent(settings)
        answer_agent = build_answer_agent(settings)
        app.state.ctx = AppState(
            settings=settings,
            source=source,
            schema_text=schema_text,
            dict_text=dict_text,
            sql_agent=sql_agent,
            answer_agent=answer_agent,
        )
        yield
    finally:
        source.close()


app = FastAPI(lifespan=lifespan)


def get_ctx(request: Request) -> AppState:
    """Return the app-wide pipeline context stashed on `app.state`."""
    return request.app.state.ctx


CtxDep = Annotated[AppState, Depends(get_ctx)]


@app.get("/health")
def health(ctx: CtxDep) -> HealthResponse:
    """Report server health and the number of loaded tables."""
    return HealthResponse(status="ok", tables=ctx.source.table_count)


@app.post("/query", response_model_exclude_none=True)
def query(req: QueryRequest, ctx: CtxDep) -> QueryResponse:
    """Answer a natural-language question against the loaded workbook."""
    try:
        sql, rows = generate_sql(
            ctx.sql_agent, req.question, ctx.schema_text, ctx.dict_text, ctx.source
        )
    except SqlGenerationError as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not produce a working query for that question. Try rephrasing it.",
        ) from exc

    answer = answer_question(ctx.answer_agent, req.question, rows)
    if req.debug:
        return QueryResponse(answer=answer, sql=sql, rows=rows)
    return QueryResponse(answer=answer)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler so unexpected errors return a clean 500 instead of a stack trace."""
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
