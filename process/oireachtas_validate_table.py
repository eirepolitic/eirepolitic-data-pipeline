from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import boto3
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.io_s3 import get_bytes
from extract.oireachtas.schemas import load_table_registry

DEFAULT_EXPECTATIONS = _REPO_ROOT / "configs/oireachtas/validation_expectations.yml"
DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_REGION = "ca-central-1"


@dataclass(frozen=True)
class ValidationResult:
    table: str
    test_name: str
    expected_result: str
    actual_result: str
    status: str
    source_url_or_api_request: str = ""
    sample_record_id: str = ""
    details: str = ""


def load_expectations(path: Path = DEFAULT_EXPECTATIONS) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("validation expectations must be a mapping")
    return payload


def validate_dataframe(
    *,
    table: str,
    df: pd.DataFrame,
    expected_columns: list[str],
    primary_key: list[str],
    rules: dict[str, Any] | None = None,
    historical_start: str = "2024-11-29",
) -> list[ValidationResult]:
    rules = rules or {}
    results: list[ValidationResult] = []

    actual_columns = list(df.columns)
    results.append(
        _result(
            table,
            "schema_exact",
            expected_columns,
            actual_columns,
            actual_columns == expected_columns,
        )
    )

    missing_pk = [column for column in primary_key if column not in df.columns]
    if missing_pk:
        results.append(
            ValidationResult(
                table=table,
                test_name="primary_key_non_null",
                expected_result="all primary-key columns present and populated",
                actual_result=f"missing columns: {missing_pk}",
                status="fail",
            )
        )
        results.append(
            ValidationResult(
                table=table,
                test_name="primary_key_unique",
                expected_result="zero duplicate primary keys",
                actual_result="not evaluated because primary-key columns are missing",
                status="fail",
            )
        )
    else:
        blank_mask = pd.Series(False, index=df.index)
        for column in primary_key:
            blank_mask |= df[column].isna() | (df[column].astype(str).str.strip() == "")
        duplicate_count = int(df.duplicated(subset=primary_key, keep=False).sum())
        results.append(
            _result(table, "primary_key_non_null", 0, int(blank_mask.sum()), not blank_mask.any())
        )
        results.append(
            _result(table, "primary_key_unique", 0, duplicate_count, duplicate_count == 0)
        )

    results.extend(_validate_dates(table=table, df=df, rules=rules, historical_start=historical_start))
    results.append(
        _result(table, "row_count_positive", "> 0", int(len(df)), len(df) > 0)
    )
    return results


def validate_csv_parquet_equivalence(
    *, table: str, csv_df: pd.DataFrame, parquet_df: pd.DataFrame
) -> list[ValidationResult]:
    results = [
        _result(
            table,
            "csv_parquet_row_count_equal",
            int(len(csv_df)),
            int(len(parquet_df)),
            len(csv_df) == len(parquet_df),
        ),
        _result(
            table,
            "csv_parquet_columns_equal",
            list(csv_df.columns),
            list(parquet_df.columns),
            list(csv_df.columns) == list(parquet_df.columns),
        ),
    ]
    return results


def _validate_dates(
    *, table: str, df: pd.DataFrame, rules: dict[str, Any], historical_start: str
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for column in rules.get("date_columns") or []:
        if column not in df.columns:
            results.append(
                ValidationResult(
                    table=table,
                    test_name=f"date_parse:{column}",
                    expected_result="column present; all nonblank values parse as dates",
                    actual_result="column missing",
                    status="fail",
                )
            )
            continue
        raw = df[column].fillna("").astype(str).str.strip()
        nonblank = raw[raw != ""]
        parsed = pd.to_datetime(nonblank, errors="coerce", utc=True)
        invalid_count = int(parsed.isna().sum())
        results.append(
            _result(table, f"date_parse:{column}", 0, invalid_count, invalid_count == 0)
        )

    coverage_column = rules.get("coverage_column")
    if coverage_column:
        if coverage_column not in df.columns:
            results.append(
                ValidationResult(
                    table=table,
                    test_name="historical_coverage",
                    expected_result=f"{coverage_column} includes records on or after {historical_start}",
                    actual_result="coverage column missing",
                    status="fail",
                )
            )
        else:
            parsed = pd.to_datetime(df[coverage_column], errors="coerce")
            valid = parsed.dropna()
            actual = valid.min().date().isoformat() if not valid.empty else "no valid dates"
            expected = pd.Timestamp(historical_start)
            # A table may legitimately begin after the requested start when no official event occurred.
            passed = not valid.empty and valid.min() >= expected
            results.append(
                ValidationResult(
                    table=table,
                    test_name="historical_coverage",
                    expected_result=f"minimum valid date is not before requested start {historical_start}; gaps require source review",
                    actual_result=actual,
                    status="pass" if passed else "fail",
                    details="Event tables are checked against official sitting/event dates in later checkpoints.",
                )
            )
    return results


def _result(table: str, test_name: str, expected: Any, actual: Any, passed: bool) -> ValidationResult:
    return ValidationResult(
        table=table,
        test_name=test_name,
        expected_result=_display(expected),
        actual_result=_display(actual),
        status="pass" if passed else "fail",
    )


def _display(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _read_live_frames(*, s3: Any, bucket: str, table: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_key = f"{prefix}/{table}.csv"
    parquet_key = csv_key.replace("/csv/", "/parquet/").replace(".csv", ".parquet")
    csv_df = pd.read_csv(io.BytesIO(get_bytes(s3, bucket=bucket, key=csv_key)), dtype=str, keep_default_na=False)
    parquet_df = pd.read_parquet(io.BytesIO(get_bytes(s3, bucket=bucket, key=parquet_key))).astype(object)
    return csv_df, parquet_df


def write_results(results: Iterable[ValidationResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    (output_dir / "table_validation.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "table_validation.csv", index=False)

    lines = ["# Oireachtas table validation", ""]
    for table, group in pd.DataFrame(rows).groupby("table", sort=True):
        lines.extend([f"## {table}", "", "| Test | Expected | Actual | Status |", "|---|---|---|---|"])
        for _, row in group.iterrows():
            lines.append(
                f"| {row['test_name']} | {_md(row['expected_result'])} | {_md(row['actual_result'])} | **{row['status']}** |"
            )
        lines.append("")
    (output_dir / "table_validation.md").write_text("\n".join(lines), encoding="utf-8")


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate live Oireachtas table schemas, keys, dates, and file equivalence.")
    p.add_argument("--table", action="append", default=[])
    p.add_argument("--expectations", type=Path, default=DEFAULT_EXPECTATIONS)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output"))
    return p


def main() -> int:
    args = parser().parse_args()
    expectations = load_expectations(args.expectations)
    registry = load_table_registry()
    requested = args.table or sorted(registry)
    unknown = sorted(set(requested) - set(registry))
    if unknown:
        raise ValueError(f"unknown tables: {unknown}")

    s3 = boto3.client("s3", region_name=args.region)
    results: list[ValidationResult] = []
    prefix = str(expectations.get("logical_prefix") or "processed/oireachtas_unified/latest/csv")
    table_rules = expectations.get("table_rules") or {}
    historical_start = str(expectations.get("historical_start") or "2024-11-29")

    for table in requested:
        schema = registry[table]
        csv_df, parquet_df = _read_live_frames(s3=s3, bucket=args.bucket, table=table, prefix=prefix)
        results.extend(
            validate_dataframe(
                table=table,
                df=csv_df,
                expected_columns=schema.columns,
                primary_key=schema.primary_key,
                rules=table_rules.get(table) or {},
                historical_start=historical_start,
            )
        )
        results.extend(validate_csv_parquet_equivalence(table=table, csv_df=csv_df, parquet_df=parquet_df))

    write_results(results, args.output_dir)
    failed = [result for result in results if result.status == "fail"]
    print(json.dumps({"tables": len(requested), "tests": len(results), "failed": len(failed)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
