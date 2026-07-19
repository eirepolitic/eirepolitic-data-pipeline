from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.io_s3 import get_bytes, get_json
from extract.oireachtas.normalize import stable_hash
from extract.oireachtas.schemas import load_table_registry
from extract.oireachtas import table_control_pipeline_runs as runs_mod
from extract.oireachtas import table_control_table_manifests as manifests_mod
from extract.oireachtas import table_control_data_quality_results as dq_mod
from process.oireachtas_validate_table import (
    ValidationResult,
    load_expectations,
    validate_csv_parquet_equivalence,
    validate_dataframe,
)

TABLES = ["control_pipeline_runs", "control_table_manifests", "control_data_quality_results"]


def make_result(table: str, test: str, expected: Any, actual: Any, passed: bool, *, details: str = "", source: str = "", sample_id: str = "") -> ValidationResult:
    return ValidationResult(
        table=table,
        test_name=test,
        expected_result=_display(expected),
        actual_result=_display(actual),
        status="pass" if passed else "fail",
        source_url_or_api_request=source,
        sample_record_id=sample_id,
        details=details,
    )


def _display(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def read_live(s3: Any, *, bucket: str, table: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_key = f"{prefix}/{table}.csv"
    parquet_key = csv_key.replace("/csv/", "/parquet/").replace(".csv", ".parquet")
    csv_df = pd.read_csv(io.BytesIO(get_bytes(s3, bucket=bucket, key=csv_key)), dtype=str, keep_default_na=False)
    parquet_df = pd.read_parquet(io.BytesIO(get_bytes(s3, bucket=bucket, key=parquet_key))).astype(object)
    return csv_df, parquet_df


def object_exists(s3: Any, *, bucket: str, key: str) -> bool:
    if not key:
        return False
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def read_object_frame(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = get_bytes(s3, bucket=bucket, key=key)
    if key.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(body))
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def validate_pipeline_runs(df: pd.DataFrame, s3: Any, *, bucket: str) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    missing_manifest: list[dict[str, str]] = []
    mismatches: list[dict[str, Any]] = []
    cadence = runs_mod._cadence_by_table()
    for _, row in df.iterrows():
        key = str(row["manifest_s3_key"])
        if not object_exists(s3, bucket=bucket, key=key):
            missing_manifest.append({"run_id": str(row["run_id"]), "key": key})
            continue
        payload = get_json(s3, bucket=bucket, key=key)
        expected = runs_mod._manifest_to_row(payload, key=key, cadence_by_table=cadence)
        differences: dict[str, Any] = {}
        for column in df.columns:
            live_value = str(row.get(column, ""))
            expected_value = str(expected.get(column, ""))
            if live_value != expected_value:
                differences[column] = {"live": live_value, "expected": expected_value}
        if differences:
            mismatches.append({"run_id": str(row["run_id"]), "differences": differences})
    results.append(make_result("control_pipeline_runs", "every_manifest_reference_exists", 0, len(missing_manifest), not missing_manifest, details=json.dumps(missing_manifest[:20], ensure_ascii=False)))
    results.append(make_result("control_pipeline_runs", "rows_recompute_from_referenced_manifest", 0, len(mismatches), not mismatches, details=json.dumps(mismatches[:20], ensure_ascii=False)))

    started = pd.to_datetime(df["started_at_utc"], errors="coerce", utc=True)
    finished = pd.to_datetime(df["finished_at_utc"], errors="coerce", utc=True)
    invalid_time = df[started.notna() & finished.notna() & (started > finished)]
    results.append(make_result("control_pipeline_runs", "started_not_after_finished", 0, len(invalid_time), invalid_time.empty, details=invalid_time.head(20).to_json(orient="records")))
    success_with_error = df[(df["status"].astype(str) == "success") & (df["error_message"].astype(str).str.strip() != "")]
    results.append(make_result("control_pipeline_runs", "successful_runs_have_no_error_message", 0, len(success_with_error), success_with_error.empty, details=success_with_error.head(20).to_json(orient="records")))
    return results


def validate_table_manifests(df: pd.DataFrame, s3: Any, *, bucket: str, registry: dict[str, Any]) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    missing_objects: list[dict[str, str]] = []
    count_mismatches: list[dict[str, Any]] = []
    schema_mismatches: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        table = str(row["table_name"])
        csv_key = str(row["latest_csv_key"])
        parquet_key = str(row["latest_parquet_key"])
        for kind, key in [("csv", csv_key), ("parquet", parquet_key)]:
            if not object_exists(s3, bucket=bucket, key=key):
                missing_objects.append({"table": table, "kind": kind, "key": key})
        if csv_key and object_exists(s3, bucket=bucket, key=csv_key):
            actual_rows = len(read_object_frame(s3, bucket=bucket, key=csv_key))
            expected_rows = int(float(str(row["row_count"]))) if str(row["row_count"]).strip() else -1
            if actual_rows != expected_rows:
                count_mismatches.append({"table": table, "expected": expected_rows, "actual_csv": actual_rows})
        if parquet_key and object_exists(s3, bucket=bucket, key=parquet_key):
            actual_rows = len(read_object_frame(s3, bucket=bucket, key=parquet_key))
            expected_rows = int(float(str(row["row_count"]))) if str(row["row_count"]).strip() else -1
            if actual_rows != expected_rows:
                count_mismatches.append({"table": table, "expected": expected_rows, "actual_parquet": actual_rows})
        schema = registry.get(table)
        if schema:
            expected_hash = stable_hash([table, ",".join(schema.primary_key), ",".join(schema.columns)], length=24)
            expected_count = str(len(schema.columns))
            differences = {}
            if str(row["schema_hash"]) != expected_hash:
                differences["schema_hash"] = {"live": str(row["schema_hash"]), "expected": expected_hash}
            if str(row["column_count"]) != expected_count:
                differences["column_count"] = {"live": str(row["column_count"]), "expected": expected_count}
            if differences:
                schema_mismatches.append({"table": table, "differences": differences})
    results.append(make_result("control_table_manifests", "referenced_csv_and_parquet_objects_exist", 0, len(missing_objects), not missing_objects, details=json.dumps(missing_objects[:30], ensure_ascii=False)))
    results.append(make_result("control_table_manifests", "manifest_row_counts_equal_actual_objects", 0, len(count_mismatches), not count_mismatches, details=json.dumps(count_mismatches[:30], ensure_ascii=False)))
    results.append(make_result("control_table_manifests", "schema_hash_and_column_count_recompute", 0, len(schema_mismatches), not schema_mismatches, details=json.dumps(schema_mismatches[:30], ensure_ascii=False)))
    return results


def manifest_index(s3: Any, *, bucket: str) -> dict[str, tuple[str, dict[str, Any]]]:
    output: dict[str, tuple[str, dict[str, Any]]] = {}
    for key in dq_mod._list_manifest_keys(s3, bucket=bucket, prefix=dq_mod.MANIFEST_PREFIX):
        try:
            payload = get_json(s3, bucket=bucket, key=key)
        except Exception:
            continue
        run_id = str(payload.get("run_id") or dq_mod._run_id_from_key(key))
        output[run_id] = (key, payload)
    return output


def validate_dq_results(df: pd.DataFrame, s3: Any, *, bucket: str) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    manifests = manifest_index(s3, bucket=bucket)
    missing_runs: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for run_id, group in df.groupby("run_id", sort=False):
        record = manifests.get(str(run_id))
        if not record:
            missing_runs.append(str(run_id))
            continue
        key, payload = record
        expected_rows = dq_mod._manifest_checks(payload, manifest_key=key, created_at_utc="ignored")
        expected = {(row["check_name"], row["dq_result_id"]): row for row in expected_rows}
        for _, row in group.iterrows():
            check_key = (str(row["check_name"]), str(row["dq_result_id"]))
            exp = expected.get(check_key)
            if not exp:
                mismatches.append({"run_id": str(run_id), "check": str(row["check_name"]), "reason": "not generated from manifest"})
                continue
            differences = {}
            for column in ["run_id", "table_name", "check_name", "status", "metric_value", "threshold", "message"]:
                if str(row[column]) != str(exp[column]):
                    differences[column] = {"live": str(row[column]), "expected": str(exp[column])}
            if differences:
                mismatches.append({"run_id": str(run_id), "check": str(row["check_name"]), "differences": differences})
    results.append(make_result("control_data_quality_results", "every_dq_run_has_manifest", 0, len(missing_runs), not missing_runs, details=json.dumps(missing_runs[:30], ensure_ascii=False)))
    results.append(make_result("control_data_quality_results", "dq_rows_recompute_from_manifest", 0, len(mismatches), not mismatches, details=json.dumps(mismatches[:30], ensure_ascii=False)))
    return results


def validate_github_runs(df: pd.DataFrame, *, token: str, repo: str, sample_count: int = 5) -> ValidationResult:
    ids = [value for value in df["workflow_run_id"].fillna("").astype(str).unique() if value.strip()]
    if not ids:
        return make_result("control_pipeline_runs", "github_workflow_run_spot_check", "nonblank IDs resolve, or no IDs recorded", "no workflow_run_id values recorded", True)
    failures: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    for run_id in sorted(ids)[:sample_count]:
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"
        response = session.get(url, timeout=30)
        if response.status_code != 200:
            failures.append({"workflow_run_id": run_id, "status_code": response.status_code})
    return make_result("control_pipeline_runs", "github_workflow_run_spot_check", 0, len(failures), not failures, source="GitHub Actions API", details=json.dumps(failures, ensure_ascii=False))


def write_report(results: list[ValidationResult], live: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in results]
    pd.DataFrame(rows).to_csv(output_dir / "checkpoint6_results.csv", index=False)
    (output_dir / "checkpoint6_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = []
    for table in TABLES:
        subset = [item for item in results if item.table == table]
        summary.append({"table": table, "live_rows": len(live[table]), "tests": len(subset), "passed": sum(item.status == "pass" for item in subset), "failed": sum(item.status == "fail" for item in subset)})
    (output_dir / "checkpoint6_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Oireachtas validation — Checkpoint 6", "", "| Table | Live rows | Tests | Passed | Failed |", "|---|---:|---:|---:|---:|"]
    for item in summary:
        lines.append(f"| {item['table']} | {item['live_rows']} | {item['tests']} | {item['passed']} | {item['failed']} |")
    failures = [item for item in results if item.status == "fail"]
    lines.extend(["", "## Findings", ""])
    if not failures:
        lines.append("No failures were found.")
    else:
        lines.extend(["| Table | Test | Expected | Actual | Details |", "|---|---|---|---|---|"])
        for item in failures:
            lines.append(f"| {item.table} | {item.test_name} | {item.expected_result} | {item.actual_result} | {item.details[:1200].replace('|', '\\|')} |")
    (output_dir / "checkpoint6_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate Oireachtas control tables against manifests, S3 objects, and GitHub runs.")
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--expectations", type=Path, default=Path("configs/oireachtas/validation_expectations.yml"))
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/checkpoint6"))
    return p


def main() -> int:
    args = parser().parse_args()
    expectations = load_expectations(args.expectations)
    prefix = str(expectations.get("logical_prefix") or "processed/oireachtas_unified/latest/csv")
    rules = expectations.get("table_rules") or {}
    registry = load_table_registry()
    s3 = boto3.client("s3", region_name=args.region)
    live: dict[str, pd.DataFrame] = {}
    results: list[ValidationResult] = []
    for table in TABLES:
        csv_df, parquet_df = read_live(s3, bucket=args.bucket, table=table, prefix=prefix)
        live[table] = csv_df
        results.extend(validate_dataframe(table=table, df=csv_df, expected_columns=registry[table].columns, primary_key=registry[table].primary_key, rules=rules.get(table) or {}, historical_start=str(expectations.get("historical_start") or "2024-11-29")))
        results.extend(validate_csv_parquet_equivalence(table=table, csv_df=csv_df, parquet_df=parquet_df))
    results.extend(validate_pipeline_runs(live["control_pipeline_runs"], s3, bucket=args.bucket))
    results.extend(validate_table_manifests(live["control_table_manifests"], s3, bucket=args.bucket, registry=registry))
    results.extend(validate_dq_results(live["control_data_quality_results"], s3, bucket=args.bucket))
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if token and repo:
        results.append(validate_github_runs(live["control_pipeline_runs"], token=token, repo=repo))
    else:
        results.append(make_result("control_pipeline_runs", "github_workflow_run_spot_check", "GitHub credentials available", "not available", False))
    write_report(results, live, args.output_dir)
    failed = [item for item in results if item.status == "fail"]
    print(json.dumps({"tables": len(TABLES), "tests": len(results), "passed": len(results) - len(failed), "failed": len(failed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
