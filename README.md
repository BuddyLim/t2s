# t2s — natural-language querying over Excel

Ask plain-English questions about an Excel workbook. `t2s` generates SQL with an
LLM (text-to-SQL), runs it against the data in DuckDB, and prints a natural-language
answer. An optional Word document can be supplied as a data dictionary; without one,
the tool runs Excel-only.

```
? which city has the most customers?
╭─ Answer ─────────────────────────────────────────╮
│ NY has the most customers (2), followed by LA (1).│
╰──────────────────────────────────────────────────╯
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
- **PydanticAI** drives the two Claude models (SQL generation + answer phrasing),
  swappable via config, including a custom `base_url`.
- Generated SQL is executed **read-only** — anything that isn't a plain `SELECT`
  is rejected before it touches the data.
- **Self-correcting:** the SQL is run during validation; if DuckDB rejects it
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

```bash
uv run t2s.py            # start the interactive prompt
uv run t2s.py --debug    # also print the generated SQL and raw rows
```

Type questions at the `?` prompt; `exit` / `quit` (or Ctrl-D) to leave.

## Development

```bash
uv run ruff check .      # lint
uv run pytest            # tests
```
