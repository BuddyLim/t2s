# t2s — natural-language querying over Excel

Ask plain-English questions about an Excel workbook over a small FastAPI HTTP
server. `t2s` generates SQL with an LLM (text-to-SQL), runs it against the data
in DuckDB, and returns a natural-language answer. An optional Word document can
be supplied as a data dictionary; without one, the server runs Excel-only.

```
$ curl -X POST localhost:8000/query \
    -H 'content-type: application/json' \
    -d '{"question":"which city has the most customers?"}'
{"answer":"NY has the most customers (2), followed by LA (1)."}
```

## How it works

```
question ─▶ agent.py (LLM → validated SQL)
                │  schema (data_source) + optional dictionary (data_dictionary)
                ▼
          data_source.py (read-only SQL on DuckDB)
                ▼
          answer.py (LLM → natural-language answer)
```

- **DuckDB** reads the `.xlsx` directly; each sheet becomes a table.
- **LangChain** (`langchain-anthropic`) drives the two Claude models (SQL
  generation + answer phrasing), swappable via config, including a custom `base_url`.
  The SQL step uses `with_structured_output` to always return a single SQL string.
- Generated SQL is executed **read-only** — anything that isn't a plain `SELECT`
  is rejected before it touches the data.
- **Self-correcting:** the SQL is run right after generation; if DuckDB rejects it
  (bad column, wrong function, syntax error) the actual error is fed back to the
  model, which retries up to 3 times before giving up.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                 # install dependencies
cp .env.example .env    # then edit .env
```

Set these in `.env`:

- `ANTHROPIC_API_KEY` — your Anthropic key (or leave blank + set `BASE_URL` for a gateway)
- `EXCEL_PATH` — path to your Excel workbook (required)
- `DICT_PATH` — optional path to a Word data dictionary; omit to run Excel-only
- `SQL_MODEL` / `ANSWER_MODEL` — optional model overrides

## Usage

Run the server:

```bash
uv run fastapi dev      # dev server (reload), uses the pyproject entrypoint
uv run fastapi run      # production server
```

Interactive API docs are served at `http://localhost:8000/docs`.

Check health:

```bash
curl localhost:8000/health
# {"status":"ok","tables":3}
```

Ask a question:

```bash
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"how many rows are there?"}'
# {"answer":"There are 42 rows."}
```

Pass `"debug":true` to also get the generated SQL and raw rows:

```bash
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"how many rows are there?","debug":true}'
# {"answer":"There are 42 rows.","sql":"SELECT COUNT(*) ...","rows":[{"count":42}]}
```

If the model can't produce a working query, `/query` returns `422` with a
`detail` message; unexpected server errors return `500`.

### Web UI

A small Streamlit front-end (`ui.py`) is available as a thin HTTP client over the
API. Install its extras, start the API, then launch the UI in another terminal:

```bash
uv sync --group ui           # install streamlit + httpx
uv run fastapi run           # terminal 1: start the API
uv run streamlit run ui.py   # terminal 2: start the UI
```

Open http://localhost:8501 to ask questions in the browser. The UI reads the
API base URL from the `T2S_API_URL` env var (default `http://localhost:8000`).

## Development

```bash
uv run ruff check .      # lint
uv run ty check          # type check
uv run pytest            # tests
```
