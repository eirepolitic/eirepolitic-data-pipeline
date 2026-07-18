from __future__ import annotations

import pandas as pd

from process.oireachtas_validate_cross_table import validate_relationship
from process.oireachtas_validate_external import compare_fields, deterministic_sample, normalize_text
from process.oireachtas_validate_table import validate_csv_parquet_equivalence, validate_dataframe


def test_validate_dataframe_passes_schema_keys_and_dates() -> None:
    df = pd.DataFrame(
        [
            {"id": "a", "event_date": "2024-12-01", "snapshot_date": "2026-07-18"},
            {"id": "b", "event_date": "2025-01-01", "snapshot_date": "2026-07-18"},
        ]
    )
    results = validate_dataframe(
        table="sample",
        df=df,
        expected_columns=["id", "event_date", "snapshot_date"],
        primary_key=["id"],
        rules={"date_columns": ["event_date", "snapshot_date"], "coverage_column": "event_date"},
        historical_start="2024-11-29",
    )
    assert results
    assert all(result.status == "pass" for result in results)


def test_validate_dataframe_detects_duplicate_key() -> None:
    df = pd.DataFrame([{"id": "a"}, {"id": "a"}])
    results = validate_dataframe(
        table="sample",
        df=df,
        expected_columns=["id"],
        primary_key=["id"],
    )
    duplicate = next(result for result in results if result.test_name == "primary_key_unique")
    assert duplicate.status == "fail"
    assert duplicate.actual_result == "2"


def test_csv_parquet_equivalence_detects_column_difference() -> None:
    csv_df = pd.DataFrame([{"id": "a", "name": "A"}])
    parquet_df = pd.DataFrame([{"id": "a"}])
    results = validate_csv_parquet_equivalence(table="sample", csv_df=csv_df, parquet_df=parquet_df)
    assert results[0].status == "pass"
    assert results[1].status == "fail"


def test_relationship_validation_detects_orphan() -> None:
    parent = pd.DataFrame([{"id": "a"}])
    child = pd.DataFrame([{"parent_id": "a"}, {"parent_id": "missing"}])
    result = validate_relationship(
        name="child_parent",
        child=child,
        parent=parent,
        child_columns=["parent_id"],
        parent_columns=["id"],
    )
    assert result.status == "fail"
    assert result.actual_result == "1"


def test_deterministic_sample_is_repeatable() -> None:
    df = pd.DataFrame([{"id": str(index)} for index in range(20)])
    first = deterministic_sample(df, key_columns=["id"], count=5)
    second = deterministic_sample(df, key_columns=["id"], count=5)
    assert first["id"].tolist() == second["id"].tolist()
    assert len(first) == 5


def test_external_compare_normalizes_whitespace_and_unicode() -> None:
    assert normalize_text("A\u00a0  B") == "A B"
    results = compare_fields(
        table="sample",
        sample_record_id="1",
        source_request="https://api.oireachtas.ie/example",
        expected={"name": "Ciarán  Ahern"},
        actual={"name": "Ciarán Ahern"},
        fields=["name"],
    )
    assert results[0].status == "pass"
