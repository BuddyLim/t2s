"""Load an Excel workbook into DuckDB and run read-only SQL against it.

Each worksheet becomes its own DuckDB table (sheet name sanitised into a valid
identifier). DuckDB reads .xlsx directly via its `excel` extension, so no pandas
or openpyxl is required. Sheet names are enumerated from the workbook's XML using
only the standard library.
"""

from __future__ import annotations

import re
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
from defusedxml.ElementTree import fromstring as _xml_fromstring

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|"
    r"attach|copy|pragma|install|load|export|import|call)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    """Raised when a generated SQL query is not a plain read-only SELECT."""


def _sanitise_table_name(name: str) -> str:
    """Turn a sheet name into a safe lowercase SQL identifier."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    if not cleaned:
        cleaned = "sheet"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def _sheet_names(excel_path: Path) -> list[str]:
    """Read worksheet names from the xlsx package without extra dependencies."""
    ns = {
        "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    }
    with zipfile.ZipFile(excel_path) as zf:
        root = _xml_fromstring(zf.read("xl/workbook.xml"))
    return [
        sheet.attrib["name"]
        for sheet in root.findall("s:sheets/s:sheet", ns)
        if "name" in sheet.attrib
    ]


def is_read_only(sql: str) -> bool:
    """Return True if `sql` is a single read-only SELECT/WITH statement."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False
    # Reject anything with multiple statements.
    if ";" in stripped:
        return False
    if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
        return False
    if _FORBIDDEN.search(stripped):
        return False
    return True


@dataclass
class DataSource:
    """An in-memory DuckDB view over an Excel workbook."""

    excel_path: Path
    _con: duckdb.DuckDBPyConnection | None = field(default=None, init=False, repr=False)
    _tables: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def load(self) -> DataSource:
        """Load every sheet of the workbook into a DuckDB table."""
        path = Path(self.excel_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

        con = duckdb.connect(database=":memory:")
        con.execute("INSTALL excel; LOAD excel;")

        used: set[str] = set()
        for sheet in _sheet_names(path):
            table = _sanitise_table_name(sheet)
            while table in used:
                table = f"{table}_x"
            used.add(table)
            con.execute(
                f'CREATE TABLE "{table}" AS '
                "SELECT * FROM read_xlsx(?, sheet = ?, all_varchar = false)",
                [str(path), sheet],
            )
            self._tables[table] = sheet

        self._con = con
        return self

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the live DuckDB connection, or raise if load() hasn't run."""
        if self._con is None:
            raise RuntimeError("DataSource.load() must be called before use.")
        return self._con

    @property
    def table_count(self) -> int:
        """Return the number of tables loaded from the workbook."""
        return len(self._tables)

    def schema_text(self) -> str:
        """Return a human/LLM-readable summary of tables and their columns."""
        lines: list[str] = []
        for table, sheet in self._tables.items():
            cols = self.connection.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
            col_desc = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            lines.append(f'Table "{table}" (from sheet "{sheet}"): {col_desc}')
        return "\n".join(lines)

    def run_sql(self, sql: str, limit: int = 1000) -> list[dict]:
        """Execute a read-only SELECT and return rows as dicts.

        Raises UnsafeQueryError if the statement is not a plain SELECT.
        """
        if not is_read_only(sql):
            raise UnsafeQueryError(f"Refusing to run non-read-only SQL: {sql!r}")
        with self._lock:
            cur = self.connection.execute(sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(limit)
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def close(self) -> None:
        """Close the underlying DuckDB connection if one is open."""
        if self._con is not None:
            self._con.close()
            self._con = None
