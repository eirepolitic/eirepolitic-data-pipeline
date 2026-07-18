from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from process.oireachtas_validate_table import DEFAULT_EXPECTATIONS, ValidationResult, load_expectations

OFFICIAL_HOSTS = {"api.oireachtas.ie", "www.oireachtas.ie", "data.oireachtas.ie", "oireachtas.ie"}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deterministic_sample(df: pd.DataFrame, *, key_columns: list[str], count: int = 5) -> pd.DataFrame:
    if df.empty or count <= 0:
        return df.head(0).copy()
    missing = [column for column in key_columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing sample key columns: {missing}")
    working = df.copy()
    working["__sample_hash"] = working[key_columns].fillna("").astype(str).agg("|".join, axis=1).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    return working.sort_values("__sample_hash").head(min(count, len(working))).drop(columns=["__sample_hash"])


def compare_fields(
    *,
    table: str,
    sample_record_id: str,
    source_request: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
    fields: Iterable[str],
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for field in fields:
        expected_value = normalize_text(expected.get(field))
        actual_value = normalize_text(actual.get(field))
        results.append(
            ValidationResult(
                table=table,
                test_name=f"external_match:{field}",
                expected_result=expected_value,
                actual_result=actual_value,
                status="pass" if expected_value == actual_value else "fail",
                source_url_or_api_request=source_request,
                sample_record_id=sample_record_id,
            )
        )
    return results


def validate_external_registry(expectations: dict[str, Any], registry_tables: set[str]) -> list[ValidationResult]:
    rules = expectations.get("table_rules") or {}
    groups = expectations.get("external_groups") or {}
    results: list[ValidationResult] = []
    grouped_tables = {table for tables in groups.values() for table in tables}
    for table in sorted(registry_tables):
        strategy = str((rules.get(table) or {}).get("external_strategy") or "")
        results.append(
            ValidationResult(
                table=table,
                test_name="external_strategy_registered",
                expected_result="nonblank official-source or independent-recompute strategy",
                actual_result=strategy or "missing",
                status="pass" if strategy else "fail",
            )
        )
        results.append(
            ValidationResult(
                table=table,
                test_name="external_group_registered",
                expected_result="table appears in exactly one external validation group",
                actual_result="registered" if table in grouped_tables else "missing",
                status="pass" if table in grouped_tables else "fail",
            )
        )
    return results


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and validate the Oireachtas external-source test matrix.")
    p.add_argument("--expectations", type=Path, default=DEFAULT_EXPECTATIONS)
    p.add_argument("--tables-config", type=Path, default=Path("configs/oireachtas/tables.yml"))
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output"))
    return p


def main() -> int:
    args = parser().parse_args()
    expectations = load_expectations(args.expectations)
    tables_payload = __import__("yaml").safe_load(args.tables_config.read_text(encoding="utf-8")) or {}
    registry_tables = set((tables_payload.get("tables") or {}).keys())
    results = validate_external_registry(expectations, registry_tables)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    (args.output_dir / "external_validation_matrix.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_dir / "external_validation_matrix.csv", index=False)

    plan = {
        table: {
            "strategy": (expectations.get("table_rules") or {}).get(table, {}).get("external_strategy"),
            "sample_policy": "earliest, latest, and deterministic hash sample",
            "official_hosts": sorted(OFFICIAL_HOSTS),
        }
        for table in sorted(registry_tables)
    }
    (args.output_dir / "external_validation_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = [result for result in results if result.status == "fail"]
    print(json.dumps({"tables": len(registry_tables), "matrix_tests": len(results), "failed": len(failed)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
