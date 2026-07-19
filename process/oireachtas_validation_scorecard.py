from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.io_s3 import get_bytes
from extract.oireachtas.schemas import load_table_registry

KNOWN_FINDINGS: dict[str, dict[str, Any]] = {
    "silver_member_parties": {
        "classification": "repair_required",
        "finding": "196 duplicate business rows across 98 members; no conflicting current party values.",
    },
    "silver_member_constituencies": {
        "classification": "repair_required",
        "finding": "196 duplicate business rows across 98 members; no conflicting current constituency values.",
    },
    "silver_debate_sections": {
        "classification": "refresh_required",
        "finding": "7 official sections added for 2026-07-16 after the live snapshot.",
    },
    "silver_questions": {
        "classification": "refresh_required",
        "finding": "1,212 official questions added for 2026-07-14 through 2026-07-16 after the live snapshot.",
    },
    "silver_bill_versions": {
        "classification": "refresh_required",
        "finding": "One newer official bill version is absent: bill 2026/1, English version C.",
    },
    "silver_bill_debates": {
        "classification": "refresh_required",
        "finding": "A small number of July 16 debate links/titles changed in the current official response; historical ID drift is also present.",
    },
    "silver_division_tallies": {
        "classification": "pass_with_warning",
        "finding": "Official API sometimes omits zero tallies; stored zero values reconcile to individual vote rows.",
    },
    "silver_speeches": {
        "classification": "pass_with_warning",
        "finding": "A valid Seanad chair speaking at a joint sitting is not present in the Dáil-only member dimension.",
    },
    "gold_content_fact_pool": {
        "classification": "pass_with_warning",
        "finding": "Some zero metrics serialize as 0.0 rather than 0; all numeric source reconciliations pass.",
    },
}

CHECKPOINT_PATHS = {
    "checkpoint2": Path("docs/oireachtas_validation/checkpoint2/checkpoint2_summary.json"),
    "checkpoint3": Path("docs/oireachtas_validation/checkpoint3/checkpoint3_summary.json"),
    "checkpoint4": Path("docs/oireachtas_validation/checkpoint4/checkpoint4_summary.json"),
    "checkpoint5": Path("docs/oireachtas_validation/checkpoint5/checkpoint5_summary.json"),
    "checkpoint6": Path("docs/oireachtas_validation/checkpoint6/checkpoint6_summary.json"),
}


def read_frame(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = get_bytes(s3, bucket=bucket, key=key)
    if key.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(body)).astype(object)
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def checkpoint_failures() -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    table_failures: dict[str, int] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, path in CHECKPOINT_PATHS.items():
        if not path.exists():
            loaded[name] = []
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded[name] = payload if isinstance(payload, list) else []
        for row in loaded[name]:
            table = str(row.get("table") or "")
            if table and table != "cross_table":
                table_failures[table] = table_failures.get(table, 0) + int(row.get("failed") or 0)
    return table_failures, loaded


def validate_table(s3: Any, *, bucket: str, table: str, schema: Any) -> dict[str, Any]:
    csv_key = f"processed/oireachtas_unified/latest/csv/{table}.csv"
    parquet_key = f"processed/oireachtas_unified/latest/parquet/{table}.parquet"
    errors: list[str] = []
    try:
        csv_df = read_frame(s3, bucket=bucket, key=csv_key)
    except Exception as exc:
        return {
            "table": table,
            "row_count": 0,
            "schema_status": "fail",
            "primary_key_status": "fail",
            "csv_parquet_status": "fail",
            "errors": [f"CSV read failed: {type(exc).__name__}: {exc}"],
        }
    try:
        parquet_df = read_frame(s3, bucket=bucket, key=parquet_key)
    except Exception as exc:
        parquet_df = pd.DataFrame()
        errors.append(f"Parquet read failed: {type(exc).__name__}: {exc}")

    schema_ok = list(csv_df.columns) == list(schema.columns)
    pk_columns = list(schema.primary_key)
    pk_present = all(column in csv_df.columns for column in pk_columns)
    if pk_present:
        blank = pd.Series(False, index=csv_df.index)
        for column in pk_columns:
            blank |= csv_df[column].fillna("").astype(str).str.strip() == ""
        pk_ok = not blank.any() and not csv_df.duplicated(subset=pk_columns).any()
    else:
        pk_ok = False
    parity_ok = not parquet_df.empty and len(csv_df) == len(parquet_df) and list(csv_df.columns) == list(parquet_df.columns)
    return {
        "table": table,
        "row_count": int(len(csv_df)),
        "schema_status": "pass" if schema_ok else "fail",
        "primary_key_status": "pass" if pk_ok else "fail",
        "csv_parquet_status": "pass" if parity_ok else "fail",
        "errors": errors,
    }


def overall_classification(base: dict[str, Any], table: str, checkpoint_failed: int) -> tuple[str, str]:
    technical_fail = any(base[field] != "pass" for field in ["schema_status", "primary_key_status", "csv_parquet_status"])
    if technical_fail:
        return "repair_required", "Live file, schema, primary-key, or CSV/Parquet validation failed."
    known = KNOWN_FINDINGS.get(table)
    if known:
        return str(known["classification"]), str(known["finding"])
    if checkpoint_failed:
        return "review_required", f"The original checkpoint recorded {checkpoint_failed} failed check(s); inspect the checkpoint report."
    return "pass", "No unresolved finding recorded."


def write_reports(scorecard: list[dict[str, Any]], checkpoint_summaries: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_scorecard.json").write_text(json.dumps(scorecard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(scorecard).to_csv(output_dir / "final_scorecard.csv", index=False)
    counts = pd.Series([row["overall"] for row in scorecard]).value_counts().to_dict()
    summary = {
        "table_count": len(scorecard),
        "classification_counts": counts,
        "all_live_files_readable": all(not row["errors"] for row in scorecard),
        "all_schemas_pass": all(row["schema_status"] == "pass" for row in scorecard),
        "all_primary_keys_pass": all(row["primary_key_status"] == "pass" for row in scorecard),
        "all_csv_parquet_pass": all(row["csv_parquet_status"] == "pass" for row in scorecard),
        "checkpoint_summaries_present": {name: bool(rows) for name, rows in checkpoint_summaries.items()},
    }
    (output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Oireachtas 31-table validation scorecard",
        "",
        "## Summary",
        "",
        f"- Tables checked: {summary['table_count']}",
        f"- All schemas pass: {summary['all_schemas_pass']}",
        f"- All primary keys pass: {summary['all_primary_keys_pass']}",
        f"- All CSV/Parquet pairs pass: {summary['all_csv_parquet_pass']}",
        f"- Classifications: {json.dumps(counts, sort_keys=True)}",
        "",
        "| Table | Rows | Schema | Primary key | CSV/Parquet | Overall | Finding |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in scorecard:
        finding = str(row["finding"]).replace("|", "\\|")
        lines.append(
            f"| {row['table']} | {row['row_count']} | {row['schema_status']} | {row['primary_key_status']} | "
            f"{row['csv_parquet_status']} | **{row['overall']}** | {finding} |"
        )
    lines.extend([
        "",
        "## Recommended disposition",
        "",
        "1. Run a current refresh to capture the identified July 14–16 proceedings and legislation additions.",
        "2. Repair member-party and member-constituency business-key deduplication before the next history refresh.",
        "3. Retain the zero-tally, joint-sitting speaker, and fact numeric-format items as documented warnings.",
        "4. Re-run all checkpoints after repairs and refresh, then compare this scorecard to the new result.",
    ])
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a final read-only validation scorecard for all Oireachtas tables.")
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/final"))
    return p


def main() -> int:
    args = parser().parse_args()
    registry = load_table_registry()
    s3 = boto3.client("s3", region_name=args.region)
    failures, summaries = checkpoint_failures()
    scorecard: list[dict[str, Any]] = []
    for table in sorted(registry):
        base = validate_table(s3, bucket=args.bucket, table=table, schema=registry[table])
        overall, finding = overall_classification(base, table, failures.get(table, 0))
        base.update({
            "checkpoint_failed_checks": failures.get(table, 0),
            "overall": overall,
            "finding": finding,
        })
        scorecard.append(base)
    write_reports(scorecard, summaries, args.output_dir)
    summary_counts = pd.Series([row["overall"] for row in scorecard]).value_counts().to_dict()
    print(json.dumps({"tables": len(scorecard), "classifications": summary_counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
