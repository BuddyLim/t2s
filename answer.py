"""Answer agent: turn SQL result rows into a natural-language answer.

Uses the stronger model. The output is a plain string — the findings phrased for
a human reader.
"""

from __future__ import annotations

import json

from pydantic_ai import Agent

from config import Settings, build_model

_INSTRUCTIONS = """\
You are a data analyst presenting findings to a non-technical colleague.

You are given the user's original question and the rows returned by a SQL query
that was run to answer it. Write a clear, concise natural-language answer.

Rules:
- Answer the question directly using the data provided.
- Do not mention SQL, tables, or that a query was run.
- If the result set is empty, say that no matching data was found.
- Include relevant numbers, but keep it brief and readable.
"""


def build_answer_agent(settings: Settings) -> Agent[None, str]:
    """Create the natural-language answer agent bound to the answer model."""
    return Agent(
        build_model(settings, settings.answer_model),
        output_type=str,
        instructions=_INSTRUCTIONS,
    )


def answer_question(
    agent: Agent[None, str],
    question: str,
    rows: list[dict],
) -> str:
    """Produce a natural-language answer for `question` from result `rows`."""
    rows_json = json.dumps(rows, default=str, ensure_ascii=False, indent=2)
    prompt = (
        f"# Question\n{question}\n\n"
        f"# Query results ({len(rows)} rows)\n{rows_json}"
    )
    result = agent.run_sync(prompt)
    return result.output.strip()
