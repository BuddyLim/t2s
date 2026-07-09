"""Tests for the DuckDB-backed data source and its read-only guard."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from data_source import DataSource, UnsafeQueryError, is_read_only

# Referencing a fixture by its name as a test argument is the pytest idiom.
# pylint: disable=redefined-outer-name


@pytest.fixture
def sample_xlsx(tmp_path: Path) -> Path:
    """Create a small .xlsx fixture using DuckDB's excel writer."""
    path = tmp_path / "sample.xlsx"
    con = duckdb.connect()
    con.execute("INSTALL excel; LOAD excel;")
    con.execute(
        "COPY (SELECT * FROM (VALUES "
        "('Alice', 30, 'NY'), ('Bob', 25, 'LA'), ('Cara', 41, 'NY')"
        ") AS t(name, age, city)) "
        f"TO '{path}' WITH (FORMAT xlsx, HEADER true)"
    )
    con.close()
    return path


class TestReadOnlyGuard:
    """Tests for the is_read_only SQL guard."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM t",
            "  select name from t where age > 20  ",
            "WITH x AS (SELECT 1) SELECT * FROM x",
        ],
    )
    def test_accepts_selects(self, sql: str) -> None:
        """Plain SELECT / CTE statements are accepted."""
        assert is_read_only(sql) is True

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE t",
            "DELETE FROM t",
            "UPDATE t SET age = 0",
            "INSERT INTO t VALUES (1)",
            "SELECT 1; DROP TABLE t",
            "CREATE TABLE x AS SELECT 1",
            "",
        ],
    )
    def test_rejects_non_selects(self, sql: str) -> None:
        """Writes, multi-statements, and empty input are rejected."""
        assert is_read_only(sql) is False


class TestDataSource:
    """Tests for loading an Excel workbook and running queries."""

    def test_loads_sheet_as_table(self, sample_xlsx: Path) -> None:
        """Each sheet is exposed as a queryable table in the schema summary."""
        source = DataSource(sample_xlsx).load()
        schema = source.schema_text()
        assert "sheet1" in schema
        assert "name" in schema and "age" in schema
        source.close()

    def test_run_sql_returns_dicts(self, sample_xlsx: Path) -> None:
        """A SELECT returns rows as column-keyed dicts."""
        source = DataSource(sample_xlsx).load()
        rows = source.run_sql("SELECT name, age FROM sheet1 WHERE city = 'NY' ORDER BY name")
        assert rows == [
            {"name": "Alice", "age": 30.0},
            {"name": "Cara", "age": 41.0},
        ]
        source.close()

    def test_run_sql_blocks_writes(self, sample_xlsx: Path) -> None:
        """run_sql refuses non-read-only statements."""
        source = DataSource(sample_xlsx).load()
        with pytest.raises(UnsafeQueryError):
            source.run_sql("DROP TABLE sheet1")
        source.close()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent workbook raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DataSource(tmp_path / "nope.xlsx").load()
