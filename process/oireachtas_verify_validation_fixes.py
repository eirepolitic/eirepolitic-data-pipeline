from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import boto3
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas import table_bill_debates as bill_debates
from extract.oireachtas import table_bill_versions as bill_versions
from extract.oireachtas import table_debate_records as debate_records
from extract.oireachtas import table_debate_sections as debate_sections
from extract.oireachtas import table_questions as questions
from extract.oireachtas.batch import (
    batch_key_for_production_key,
    batch_manifest_key,
    read_json_required,
    resolve_production_key,
)
from extract.oireachtas.client import OireachtasClient
from extract.oireachtas.normalize import stable_hash
from extract.oireachtas.partitioned_fetch import get_date_partitioned_json_summary
from extract.oireachtas.schemas import load_table_registry


LOGICAL_PREFIX = "processed/oireachtas_unified/latest"


def read_csv_direct(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def candidate_key(batch_id: str, table: str, fmt: str = "csv") -> str:
    logical = f"{LOGICAL_PREFIX}/{fmt}/{table}.{fmt if fmt == 'csv' else 'parquet'}"
    return batch_key_for_production_key(logical, batch_id)


def read_candidate(s3: Any, *, bucket: str, batch_id: str, table: str) -> pd.DataFrame:
    return read_csv_direct(s3, bucket=bucket, key=candidate_key(batch_id, table, "csv"))


def read_production(s3: Any, *, bucket: str, table: str) -> pd.DataFrame:
    logical = f"{LOGICAL_PREFIX}/csv/{table}.csv"
    return read_csv_direct(s3, bucket=bucket, key=resolve_production_key(s3, bucket=bucket, production_key=logical))


def truthy(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def business_duplicates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[df.duplicated(subset=columns, keep=False)].copy()


def current_value_sets(df: pd.DataFrame, *, value_columns: list[str]) -> dict[str, set[tuple[str, ...]]]:
    current = df[truthy(df["is_current"])].copy()
    output: dict[str, set[tuple[str, ...]]] = {}
    for member_code, group in current.groupby("member_code", dropna=False):
        output[str(member_code)] = {
            tuple(str(row.get(column, "")).strip() for column in value_columns)
            for _, row in group.iterrows()
        }
    return output


def compare_current_values(
    candidate: pd.DataFrame,
    production: pd.DataFrame,
    *,
    value_columns: list[str],
) -> list[dict[str, Any]]:
    left = current_value_sets(candidate, value_columns=value_columns)
    right = current_value_sets(production, value_columns=value_columns)
    differences: list[dict[str, Any]] = []
    for member_code in sorted(set(left) | set(right)):
        if left.get(member_code, set()) != right.get(member_code, set()):
            differences.append(
                {
                    "member_code": member_code,
                    "candidate": sorted(left.get(member_code, set())),
                    "production": sorted(right.get(member_code, set())),
                }
            )
    return differences


def dedupe_frame(rows: Iterable[Mapping[str, Any]], key: str) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty or key not in frame.columns:
        return frame
    return frame.drop_duplicates(subset=[key], keep="first")


def fetch_official_recent(client: OireachtasClient, *, date_end: str) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    snapshot = date_end
    debate_summary = client.get_json_summary(
        "/debates",
        params={
            "chamber_id": "/ie/oireachtas/house/dail/34",
            "lang": "en",
            "date_start": "2026-07-14",
            "date_end": date_end,
            "limit": 200,
        },
    )
    question_summary = get_date_partitioned_json_summary(
        client,
        "/questions",
        params={
            "chamber": "dail",
            "house_no": "34",
            "date_start": "2026-07-14",
            "date_end": date_end,
            "limit": 200,
        },
    )
    legislation_summary = client.get_json_summary(
        "/legislation",
        params={
            "chamber": "dail",
            "house_no": "34",
            "date_start": "2024-11-29",
            "date_end": date_end,
            "limit": 200,
        },
    )
    for name, summary in {
        "debates": debate_summary,
        "questions": question_summary,
        "legislation": legislation_summary,
    }.items():
        if not summary.ok or not summary.payload:
            raise RuntimeError(f"official {name} API failed: {summary.error or summary.status_code}")

    section_rows: list[dict[str, Any]] = []
    for item in debate_summary.payload.get("results", []):
        if not isinstance(item, Mapping):
            continue
        record = item.get("debateRecord") if isinstance(item.get("debateRecord"), Mapping) else item
        debate_id = debate_records._first_text(record, "uri", "debateUri")
        section_items = record.get("debateSections") if isinstance(record.get("debateSections"), list) else []
        for index, section_item in enumerate(section_items, start=1):
            if not isinstance(section_item, Mapping) or not debate_id:
                continue
            section = section_item.get("debateSection") if isinstance(section_item.get("debateSection"), Mapping) else section_item
            section_rows.append(
                debate_sections._normalise_section_row(
                    section,
                    debate_id=debate_id,
                    section_order=index,
                    snapshot_date=snapshot,
                )
            )

    question_rows = [
        questions._normalise_question_row(item, snapshot_date=snapshot)
        for item in question_summary.payload.get("results", [])
        if isinstance(item, Mapping)
    ]

    version_rows: list[dict[str, Any]] = []
    debate_link_rows: list[dict[str, Any]] = []
    for item in legislation_summary.payload.get("results", []):
        if not isinstance(item, Mapping):
            continue
        version_rows.extend(bill_versions._normalise_version_rows(item, snapshot_date=snapshot))
        debate_link_rows.extend(bill_debates._normalise_debate_rows(item, snapshot_date=snapshot))

    return (
        {
            "silver_debate_sections": dedupe_frame(section_rows, "debate_section_id"),
            "silver_questions": dedupe_frame(question_rows, "question_id"),
            "silver_bill_versions": dedupe_frame(version_rows, "bill_version_id"),
            "silver_bill_debates": dedupe_frame(debate_link_rows, "bill_debate_id"),
        },
        {
            "debates": debate_summary.url,
            "questions": question_summary.url,
            "legislation": legislation_summary.url,
        },
    )


def key_completeness(candidate: pd.DataFrame, official: pd.DataFrame, key: str) -> dict[str, Any]:
    candidate_keys = set(candidate[key].fillna("").astype(str))
    official_keys = set(official[key].fillna("").astype(str)) - {""}
    missing = sorted(official_keys - candidate_keys)
    return {
        "official_rows": int(len(official)),
        "candidate_rows": int(len(candidate)),
        "missing_count": len(missing),
        "missing_samples": missing[:30],
    }


def bill_debate_business_completeness(candidate: pd.DataFrame, official: pd.DataFrame, live_bill_ids: set[str]) -> dict[str, Any]:
    fields = ["bill_id", "debate_id", "debate_section_id", "debate_show_as", "debate_date", "chamber_uri"]
    scoped = official[official["bill_id"].fillna("").astype(str).isin(live_bill_ids)].copy()
    for frame in (candidate, scoped):
        frame["__key"] = frame[fields].fillna("").astype(str).agg("|".join, axis=1)
    candidate_keys = set(candidate["__key"].astype(str))
    official_keys = set(scoped["__key"].astype(str))
    missing = sorted(official_keys - candidate_keys)
    return {
        "official_rows_for_candidate_bills": int(len(scoped)),
        "candidate_rows": int(len(candidate)),
        "missing_business_rows": len(missing),
        "missing_samples": missing[:30],
    }


def validate_control_manifests(
    s3: Any,
    *,
    bucket: str,
    batch_id: str,
    manifests: pd.DataFrame,
) -> dict[str, Any]:
    registry = load_table_registry()
    missing_tables = sorted(set(registry) - set(manifests["table_name"].astype(str)))
    failures: list[dict[str, Any]] = []
    for _, row in manifests.iterrows():
        table = str(row["table_name"])
        schema = registry.get(table)
        if not schema:
            continue
        expected_hash = stable_hash([table, ",".join(schema.primary_key), ",".join(schema.columns)], length=24)
        expected_rows = int(float(str(row["row_count"]))) if str(row["row_count"]).strip() else -1
        actual_csv = len(read_csv_direct(s3, bucket=bucket, key=candidate_key(batch_id, table, "csv")))
        parquet_body = s3.get_object(Bucket=bucket, Key=candidate_key(batch_id, table, "parquet"))["Body"].read()
        actual_parquet = len(pd.read_parquet(io.BytesIO(parquet_body)))
        differences: dict[str, Any] = {}
        if expected_rows != actual_csv or expected_rows != actual_parquet:
            differences["row_count"] = {
                "stored": expected_rows,
                "actual_csv": actual_csv,
                "actual_parquet": actual_parquet,
            }
        if str(row["column_count"]) != str(len(schema.columns)):
            differences["column_count"] = {
                "stored": str(row["column_count"]),
                "expected": len(schema.columns),
            }
        if str(row["schema_hash"]) != expected_hash:
            differences["schema_hash"] = {
                "stored": str(row["schema_hash"]),
                "expected": expected_hash,
            }
        if differences:
            failures.append({"table": table, "differences": differences})
    return {
        "row_count": int(len(manifests)),
        "missing_tables": missing_tables,
        "failure_count": len(failures),
        "failure_samples": failures[:30],
    }


def check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"check": name, "status": "pass" if passed else "fail", "details": details}


def write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Oireachtas validation-fixes candidate acceptance",
        "",
        f"- Batch: `{payload['batch_id']}`",
        f"- Overall: **{payload['status']}**",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for item in payload["checks"]:
        details = json.dumps(item["details"], ensure_ascii=False, sort_keys=True).replace("|", "\\|")
        lines.append(f"| {item['check']} | **{item['status']}** | {details[:1500]} |")
    (output_dir / "acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify the repaired Oireachtas candidate against issue-specific acceptance gates.")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--date-end", default="2026-07-19")
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/fixes_acceptance"))
    return p


def main() -> int:
    args = parser().parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    batch_manifest = read_json_required(s3, bucket=args.bucket, key=batch_manifest_key(args.batch_id))

    candidate_parties = read_candidate(s3, bucket=args.bucket, batch_id=args.batch_id, table="silver_member_parties")
    candidate_constituencies = read_candidate(s3, bucket=args.bucket, batch_id=args.batch_id, table="silver_member_constituencies")
    production_parties = read_production(s3, bucket=args.bucket, table="silver_member_parties")
    production_constituencies = read_production(s3, bucket=args.bucket, table="silver_member_constituencies")

    party_duplicates = business_duplicates(candidate_parties, ["member_code", "party_uri", "party_start", "party_end"])
    constituency_duplicates = business_duplicates(candidate_constituencies, ["member_code", "constituency_uri", "represent_start", "represent_end"])
    party_current_changes = compare_current_values(candidate_parties, production_parties, value_columns=["party_uri", "party_name"])
    constituency_current_changes = compare_current_values(candidate_constituencies, production_constituencies, value_columns=["constituency_uri", "constituency_name"])

    official, sources = fetch_official_recent(
        OireachtasClient(timeout_seconds=45, retries=5, backoff_seconds=2.0, sleep_seconds=0.1),
        date_end=args.date_end,
    )
    candidate_sections = read_candidate(s3, bucket=args.bucket, batch_id=args.batch_id, table="silver_debate_sections")
    candidate_questions = read_candidate(s3, bucket=args.bucket, batch_id=args.batch_id, table="silver_questions")
    candidate_versions = read_candidate(s3, bucket=args.bucket, batch_id=args.batch_id, table="silver_bill_versions")
    candidate_bill_debates = read_candidate(s3, bucket=args.bucket, batch_id=args.batch_id, table="silver_bill_debates")
    candidate_bills = read_candidate(s3, bucket=args.bucket, batch_id=args.batch_id, table="silver_bills")
    candidate_manifests = read_candidate(s3, bucket=args.bucket, batch_id=args.batch_id, table="control_table_manifests")

    section_check = key_completeness(candidate_sections, official["silver_debate_sections"], "debate_section_id")
    question_check = key_completeness(candidate_questions, official["silver_questions"], "question_id")
    scoped_versions = official["silver_bill_versions"][
        official["silver_bill_versions"]["bill_id"].fillna("").astype(str).isin(set(candidate_bills["bill_id"].astype(str)))
    ]
    version_check = key_completeness(candidate_versions, scoped_versions, "bill_version_id")
    debate_check = bill_debate_business_completeness(
        candidate_bill_debates,
        official["silver_bill_debates"],
        set(candidate_bills["bill_id"].astype(str)),
    )
    control_check = validate_control_manifests(
        s3,
        bucket=args.bucket,
        batch_id=args.batch_id,
        manifests=candidate_manifests,
    )

    checks = [
        check(
            "batch_manifest_validated",
            batch_manifest.get("status") == "validated"
            and int(batch_manifest.get("table_count") or 0) >= 31
            and not any(batch_manifest.get("validation", {}).get(name) for name in ["missing_tables", "failed_tables", "missing_objects", "duplicate_tables"]),
            {
                "status": batch_manifest.get("status"),
                "table_count": batch_manifest.get("table_count"),
                "validation": batch_manifest.get("validation"),
            },
        ),
        check("member_party_business_keys_unique", party_duplicates.empty, {"duplicate_rows": len(party_duplicates)}),
        check("member_constituency_business_keys_unique", constituency_duplicates.empty, {"duplicate_rows": len(constituency_duplicates)}),
        check("current_party_values_unchanged", not party_current_changes, {"difference_count": len(party_current_changes), "samples": party_current_changes[:20]}),
        check("current_constituency_values_unchanged", not constituency_current_changes, {"difference_count": len(constituency_current_changes), "samples": constituency_current_changes[:20]}),
        check("recent_official_debate_sections_present", section_check["missing_count"] == 0, {**section_check, "source": sources["debates"]}),
        check("recent_official_questions_present", question_check["missing_count"] == 0, {**question_check, "source": sources["questions"]}),
        check("official_bill_versions_present", version_check["missing_count"] == 0, {**version_check, "source": sources["legislation"]}),
        check("official_bill_debate_business_rows_present", debate_check["missing_business_rows"] == 0, {**debate_check, "source": sources["legislation"]}),
        check("control_manifest_counts_and_schemas_match_candidate", not control_check["missing_tables"] and control_check["failure_count"] == 0, control_check),
    ]
    payload = {
        "batch_id": args.batch_id,
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
    }
    write_report(payload, args.output_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
