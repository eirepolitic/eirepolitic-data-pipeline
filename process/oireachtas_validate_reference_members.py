from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import boto3
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.client import OireachtasClient
from extract.oireachtas.io_s3 import get_bytes
from extract.oireachtas.schemas import load_table_registry
from extract.oireachtas import table_constituencies as constituencies
from extract.oireachtas import table_houses as houses
from extract.oireachtas import table_member_constituencies as member_constituencies
from extract.oireachtas import table_member_memberships as member_memberships
from extract.oireachtas import table_member_offices as member_offices
from extract.oireachtas import table_member_parties as member_parties
from extract.oireachtas import table_members as members
from extract.oireachtas import table_parties as parties
from process.oireachtas_validate_external import deterministic_sample, normalize_text
from process.oireachtas_validate_table import (
    ValidationResult,
    load_expectations,
    validate_csv_parquet_equivalence,
    validate_dataframe,
)

TABLES = [
    "silver_houses",
    "silver_constituencies",
    "silver_parties",
    "silver_members",
    "silver_member_memberships",
    "silver_member_parties",
    "silver_member_constituencies",
    "silver_member_offices",
]

KEYS = {
    "silver_houses": "house_uri",
    "silver_constituencies": "constituency_uri",
    "silver_parties": "party_uri",
    "silver_members": "member_code",
    "silver_member_memberships": "membership_id",
    "silver_member_parties": "member_party_id",
    "silver_member_constituencies": "member_constituency_id",
    "silver_member_offices": "member_office_id",
}

EXTERNAL_FIELDS = {
    "silver_houses": ["house_uri", "house_no", "house_code", "chamber", "show_as", "date_start", "date_end"],
    "silver_constituencies": ["constituency_uri", "constituency_code", "constituency_name", "show_as", "house_uri", "house_no", "chamber", "date_start", "date_end"],
    "silver_parties": ["party_uri", "party_code", "party_name", "show_as", "date_start", "date_end"],
    "silver_members": ["member_code", "member_uri", "full_name", "first_name", "last_name", "display_name", "gender", "latest_party_name", "latest_constituency_name", "latest_house_no"],
    "silver_member_memberships": ["membership_id", "member_code", "member_uri", "house_uri", "house_no", "house_code", "chamber", "membership_start", "membership_end"],
    "silver_member_parties": ["member_party_id", "membership_id", "member_code", "party_uri", "party_name", "party_start", "party_end"],
    "silver_member_constituencies": ["member_constituency_id", "membership_id", "member_code", "constituency_uri", "constituency_name", "represent_start", "represent_end"],
    "silver_member_offices": ["member_office_id", "membership_id", "member_code", "office_uri", "office_name", "office_start", "office_end"],
}

DATE_RANGES = {
    "silver_houses": ("date_start", "date_end"),
    "silver_constituencies": ("date_start", "date_end"),
    "silver_parties": ("date_start", "date_end"),
    "silver_member_memberships": ("membership_start", "membership_end"),
    "silver_member_parties": ("party_start", "party_end"),
    "silver_member_constituencies": ("represent_start", "represent_end"),
    "silver_member_offices": ("office_start", "office_end"),
}


def result(table: str, test: str, expected: Any, actual: Any, passed: bool, *, details: str = "", source: str = "", sample_id: str = "") -> ValidationResult:
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


def read_live(*, s3: Any, bucket: str, table: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_key = f"{prefix}/{table}.csv"
    parquet_key = csv_key.replace("/csv/", "/parquet/").replace(".csv", ".parquet")
    csv_df = pd.read_csv(io.BytesIO(get_bytes(s3, bucket=bucket, key=csv_key)), dtype=str, keep_default_na=False)
    parquet_df = pd.read_parquet(io.BytesIO(get_bytes(s3, bucket=bucket, key=parquet_key))).astype(object)
    return csv_df, parquet_df


def dedupe(rows: Iterable[dict[str, Any]], key: str) -> pd.DataFrame:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        value = str(row.get(key) or "")
        if value not in seen:
            seen.add(value)
            output.append(row)
    return pd.DataFrame(output)


def fetch_official(client: OireachtasClient) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    snapshot = date.today().isoformat()
    calls = {
        "houses": client.get_json_summary("/houses", params={"limit": 200}),
        "constituencies": client.get_json_summary("/constituencies", params={"limit": 200, "chamber": "dail", "house_no": "34"}),
        "parties": client.get_json_summary("/parties", params={"limit": 200, "chamber": "dail", "house_no": "34"}),
        "members": client.get_json_summary("/members", params={"limit": 200, "chamber": "dail", "house_no": "34"}),
    }
    for name, summary in calls.items():
        if not summary.ok or not summary.payload:
            raise RuntimeError(f"official API call failed for {name}: {summary.error or summary.status_code}")

    house_rows = [houses._normalise_house_row(item, snapshot_date=snapshot, endpoint="/houses") for item in calls["houses"].payload.get("results", [])]

    constituency_rows: list[dict[str, Any]] = []
    for item in calls["constituencies"].payload.get("results", []):
        constituency_rows.extend(
            constituencies._normalise_constituency_record(record, house, snapshot_date=snapshot, endpoint="/constituencies")
            for record, house in constituencies._iter_constituency_records(item)
        )

    party_rows: list[dict[str, Any]] = []
    for item in calls["parties"].payload.get("results", []):
        party_rows.extend(
            parties._normalise_party_record(record, snapshot_date=snapshot, endpoint="/parties")
            for record in parties._iter_party_records(item)
        )

    member_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    member_party_rows: list[dict[str, Any]] = []
    member_constituency_rows: list[dict[str, Any]] = []
    member_office_rows: list[dict[str, Any]] = []
    for item in calls["members"].payload.get("results", []):
        for member in members._iter_member_records(item):
            member_rows.append(members._normalise_member_record(member, item, snapshot_date=snapshot, endpoint="/members"))
        for member in member_memberships._iter_member_records(item):
            for membership in member_memberships._iter_memberships(member, item):
                membership_rows.append(member_memberships._normalise_membership_row(member, membership, snapshot_date=snapshot))
        for member in member_parties._iter_member_records(item):
            for membership in member_parties._iter_memberships(member, item):
                for party in member_parties._iter_parties(membership):
                    member_party_rows.append(member_parties._normalise_party_row(member, membership, party, snapshot_date=snapshot))
        for member in member_constituencies._iter_member_records(item):
            for membership in member_constituencies._iter_memberships(member, item):
                for represent in member_constituencies._iter_represents(membership):
                    member_constituency_rows.append(member_constituencies._normalise_constituency_row(member, membership, represent, snapshot_date=snapshot))
        for member in member_offices._iter_member_records(item):
            for membership in member_offices._iter_memberships(member, item):
                for office in member_offices._iter_offices(membership):
                    member_office_rows.append(member_offices._normalise_office_row(member, membership, office, snapshot_date=snapshot))

    frames = {
        "silver_houses": dedupe(house_rows, KEYS["silver_houses"]),
        "silver_constituencies": dedupe(constituency_rows, KEYS["silver_constituencies"]),
        "silver_parties": dedupe(party_rows, KEYS["silver_parties"]),
        "silver_members": dedupe(member_rows, KEYS["silver_members"]),
        "silver_member_memberships": dedupe(membership_rows, KEYS["silver_member_memberships"]),
        "silver_member_parties": dedupe(member_party_rows, KEYS["silver_member_parties"]),
        "silver_member_constituencies": dedupe(member_constituency_rows, KEYS["silver_member_constituencies"]),
        "silver_member_offices": dedupe(member_office_rows, KEYS["silver_member_offices"]),
    }
    sources = {
        "silver_houses": calls["houses"].url,
        "silver_constituencies": calls["constituencies"].url,
        "silver_parties": calls["parties"].url,
        "silver_members": calls["members"].url,
        "silver_member_memberships": calls["members"].url,
        "silver_member_parties": calls["members"].url,
        "silver_member_constituencies": calls["members"].url,
        "silver_member_offices": calls["members"].url,
    }
    return frames, sources


def truthy(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes", "y"})


def validate_range(table: str, df: pd.DataFrame, start_col: str, end_col: str) -> ValidationResult:
    start = pd.to_datetime(df[start_col], errors="coerce")
    end = pd.to_datetime(df[end_col], errors="coerce")
    invalid = df[start.notna() & end.notna() & (start > end)]
    return result(table, "date_range_order", 0, len(invalid), invalid.empty, details=invalid.head(5).to_json(orient="records"))


def validate_orphans(name: str, child: pd.DataFrame, child_col: str, parent: pd.DataFrame, parent_col: str) -> ValidationResult:
    parent_values = set(parent[parent_col].fillna("").astype(str))
    values = child[child_col].fillna("").astype(str)
    invalid = child[(values != "") & ~values.isin(parent_values)]
    return result("cross_table", name, 0, len(invalid), invalid.empty, details=invalid.head(5).to_json(orient="records"))


def validate_current_unique(table: str, df: pd.DataFrame) -> ValidationResult:
    current = df[truthy(df["is_current"])].copy()
    counts = current.groupby("member_code").size() if not current.empty else pd.Series(dtype=int)
    invalid = counts[counts > 1]
    return result(table, "at_most_one_current_row_per_member", 0, len(invalid), invalid.empty, details=invalid.head(10).to_json())


def select_current(df: pd.DataFrame, value_col: str, start_col: str, end_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["member_code", value_col])
    working = df.copy()
    working["__current"] = truthy(working["is_current"]).astype(int)
    working["__start"] = pd.to_datetime(working[start_col], errors="coerce")
    working["__end"] = pd.to_datetime(working[end_col], errors="coerce").fillna(pd.Timestamp.max)
    working = working.sort_values(["member_code", "__current", "__end", "__start"], ascending=[True, False, False, False])
    return working.drop_duplicates("member_code")[["member_code", value_col]]


def validate_member_latest(live: dict[str, pd.DataFrame]) -> list[ValidationResult]:
    output: list[ValidationResult] = []
    members_df = live["silver_members"]
    mappings = [
        ("latest_party_matches_bridge", "latest_party_name", live["silver_member_parties"], "party_name", "party_start", "party_end"),
        ("latest_constituency_matches_bridge", "latest_constituency_name", live["silver_member_constituencies"], "constituency_name", "represent_start", "represent_end"),
        ("latest_house_matches_membership", "latest_house_no", live["silver_member_memberships"], "house_no", "membership_start", "membership_end"),
    ]
    for test, member_col, bridge, value_col, start_col, end_col in mappings:
        selected = select_current(bridge, value_col, start_col, end_col)
        merged = members_df[["member_code", member_col]].merge(selected, on="member_code", how="left", suffixes=("_member", "_bridge"))
        left = merged[f"{member_col}_member"].fillna("").astype(str).map(normalize_text)
        right = merged[f"{value_col}_bridge"].fillna("").astype(str).map(normalize_text)
        evaluated = (left != "") | (right != "")
        invalid = merged[evaluated & (left != right)]
        output.append(result("silver_members", test, 0, len(invalid), invalid.empty, details=invalid.head(10).to_json(orient="records")))

    memberships = live["silver_member_memberships"]
    current_codes = set(memberships.loc[truthy(memberships["is_current"]), "member_code"].astype(str))
    flags = members_df["is_current_member"].fillna("").astype(str).str.lower().isin({"true", "1", "yes", "y"})
    expected_flags = members_df["member_code"].astype(str).isin(current_codes)
    invalid = members_df[flags != expected_flags]
    output.append(result("silver_members", "current_flag_matches_membership", 0, len(invalid), invalid.empty, details=invalid.head(10).to_json(orient="records")))
    return output


def validate_external(live: dict[str, pd.DataFrame], official: dict[str, pd.DataFrame], sources: dict[str, str], sample_count: int) -> list[ValidationResult]:
    output: list[ValidationResult] = []
    for table in TABLES:
        key = KEYS[table]
        live_df = live[table].copy()
        official_df = official[table].copy()
        live_keys = set(live_df[key].fillna("").astype(str))
        official_keys = set(official_df[key].fillna("").astype(str)) if key in official_df.columns else set()
        overlap = sorted((live_keys & official_keys) - {""})
        output.append(result(table, "official_api_key_overlap", "> 0", len(overlap), len(overlap) > 0, source=sources[table]))
        if not overlap:
            continue
        candidates = live_df[live_df[key].astype(str).isin(overlap)]
        sample = deterministic_sample(candidates, key_columns=[key], count=sample_count)
        official_index = official_df.set_index(key, drop=False)
        for _, live_row in sample.iterrows():
            record_id = str(live_row[key])
            official_row = official_index.loc[record_id]
            if isinstance(official_row, pd.DataFrame):
                official_row = official_row.iloc[0]
            mismatches: dict[str, dict[str, str]] = {}
            for field in EXTERNAL_FIELDS[table]:
                left = normalize_text(live_row.get(field, ""))
                right = normalize_text(official_row.get(field, ""))
                if left != right:
                    mismatches[field] = {"live": left, "official": right}
            output.append(
                result(
                    table,
                    "official_api_sample_match",
                    "all selected stable fields equal",
                    "match" if not mismatches else "mismatch",
                    not mismatches,
                    source=sources[table],
                    sample_id=record_id,
                    details=json.dumps(mismatches, ensure_ascii=False, sort_keys=True),
                )
            )
    return output


def write_report(results: list[ValidationResult], live: dict[str, pd.DataFrame], official: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in results]
    pd.DataFrame(rows).to_csv(output_dir / "checkpoint2_results.csv", index=False)
    (output_dir / "checkpoint2_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = []
    for table in TABLES + ["cross_table"]:
        subset = [item for item in results if item.table == table]
        if not subset:
            continue
        summary.append({
            "table": table,
            "live_rows": len(live.get(table, pd.DataFrame())),
            "official_rows": len(official.get(table, pd.DataFrame())),
            "tests": len(subset),
            "passed": sum(item.status == "pass" for item in subset),
            "failed": sum(item.status == "fail" for item in subset),
        })
    (output_dir / "checkpoint2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# Oireachtas validation — Checkpoint 2", "", "## Scorecard", "", "| Table | Live rows | Fresh official rows | Tests | Passed | Failed |", "|---|---:|---:|---:|---:|---:|"]
    for item in summary:
        lines.append(f"| {item['table']} | {item['live_rows']} | {item['official_rows']} | {item['tests']} | {item['passed']} | {item['failed']} |")
    failures = [item for item in results if item.status == "fail"]
    lines.extend(["", "## Findings", ""])
    if not failures:
        lines.append("No failures were found.")
    else:
        lines.extend(["| Table | Test | Expected | Actual | Sample | Details |", "|---|---|---|---|---|---|"])
        for item in failures:
            details = item.details.replace("|", "\\|").replace("\n", " ")[:800]
            lines.append(f"| {item.table} | {item.test_name} | {item.expected_result} | {item.actual_result} | {item.sample_record_id} | {details} |")
    lines.extend(["", "## Official API sample checks", ""])
    for item in results:
        if item.test_name == "official_api_sample_match":
            lines.append(f"- `{item.table}` / `{item.sample_record_id}`: **{item.status}** — {item.source_url_or_api_request}")
    (output_dir / "checkpoint2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate Oireachtas reference/member tables against live files and fresh official API data.")
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--expectations", type=Path, default=Path("configs/oireachtas/validation_expectations.yml"))
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/checkpoint2"))
    p.add_argument("--sample-count", type=int, default=5)
    p.add_argument("--fail-on-findings", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    expectations = load_expectations(args.expectations)
    registry = load_table_registry()
    rules = expectations.get("table_rules") or {}
    prefix = str(expectations.get("logical_prefix") or "processed/oireachtas_unified/latest/csv")
    s3 = boto3.client("s3", region_name=args.region)

    live: dict[str, pd.DataFrame] = {}
    results: list[ValidationResult] = []
    for table in TABLES:
        csv_df, parquet_df = read_live(s3=s3, bucket=args.bucket, table=table, prefix=prefix)
        live[table] = csv_df
        results.extend(validate_dataframe(table=table, df=csv_df, expected_columns=registry[table].columns, primary_key=registry[table].primary_key, rules=rules.get(table) or {}, historical_start=str(expectations.get("historical_start") or "2024-11-29")))
        results.extend(validate_csv_parquet_equivalence(table=table, csv_df=csv_df, parquet_df=parquet_df))
        if table in DATE_RANGES:
            results.append(validate_range(table, csv_df, *DATE_RANGES[table]))

    results.extend([
        validate_orphans("membership_member", live["silver_member_memberships"], "member_code", live["silver_members"], "member_code"),
        validate_orphans("membership_house", live["silver_member_memberships"], "house_uri", live["silver_houses"], "house_uri"),
        validate_orphans("member_party_member", live["silver_member_parties"], "member_code", live["silver_members"], "member_code"),
        validate_orphans("member_party_membership", live["silver_member_parties"], "membership_id", live["silver_member_memberships"], "membership_id"),
        validate_orphans("member_constituency_member", live["silver_member_constituencies"], "member_code", live["silver_members"], "member_code"),
        validate_orphans("member_constituency_membership", live["silver_member_constituencies"], "membership_id", live["silver_member_memberships"], "membership_id"),
        validate_orphans("member_office_member", live["silver_member_offices"], "member_code", live["silver_members"], "member_code"),
        validate_orphans("member_office_membership", live["silver_member_offices"], "membership_id", live["silver_member_memberships"], "membership_id"),
    ])
    for table in ["silver_member_memberships", "silver_member_parties", "silver_member_constituencies", "silver_member_offices"]:
        results.append(validate_current_unique(table, live[table]))
    results.extend(validate_member_latest(live))

    official, sources = fetch_official(OireachtasClient(timeout_seconds=45, retries=5, backoff_seconds=2.0, sleep_seconds=0.1))
    results.extend(validate_external(live, official, sources, max(1, args.sample_count)))
    write_report(results, live, official, args.output_dir)

    failed = [item for item in results if item.status == "fail"]
    print(json.dumps({"tables": len(TABLES), "tests": len(results), "passed": len(results) - len(failed), "failed": len(failed)}, indent=2))
    return 1 if args.fail_on_findings and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
