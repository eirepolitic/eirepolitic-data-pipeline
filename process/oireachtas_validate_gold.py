from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.io_s3 import get_bytes
from extract.oireachtas.schemas import load_table_registry
from extract.oireachtas import table_gold_current_members as current_mod
from extract.oireachtas import table_gold_member_activity_yearly as yearly_mod
from extract.oireachtas import table_gold_member_activity_monthly as monthly_mod
from extract.oireachtas import table_gold_constituency_activity_yearly as constituency_mod
from extract.oireachtas import table_gold_content_fact_pool as fact_mod
from process.oireachtas_validate_table import (
    ValidationResult,
    load_expectations,
    validate_csv_parquet_equivalence,
    validate_dataframe,
)

GOLD_TABLES = [
    "gold_current_members",
    "gold_member_activity_yearly",
    "gold_member_activity_monthly",
    "gold_constituency_activity_yearly",
    "gold_content_fact_pool",
]


def result(table: str, test: str, expected: Any, actual: Any, passed: bool, *, details: str = "") -> ValidationResult:
    return ValidationResult(
        table=table,
        test_name=test,
        expected_result=_display(expected),
        actual_result=_display(actual),
        status="pass" if passed else "fail",
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


def normalize_frame(df: pd.DataFrame, columns: list[str], primary_key: list[str]) -> pd.DataFrame:
    working = df.reindex(columns=columns).copy()
    for column in working.columns:
        if column == "snapshot_date":
            continue
        working[column] = working[column].fillna("").astype(str)
    sort_columns = [column for column in primary_key if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns)
    return working.reset_index(drop=True)


def compare_recomputed(
    table: str,
    live: pd.DataFrame,
    rebuilt: pd.DataFrame,
    columns: list[str],
    primary_key: list[str],
) -> list[ValidationResult]:
    compare_columns = [column for column in columns if column != "snapshot_date"]
    left = normalize_frame(live, compare_columns, primary_key)
    right = normalize_frame(rebuilt, compare_columns, primary_key)
    output = [result(table, "recomputed_row_count", len(right), len(left), len(left) == len(right))]
    if len(left) != len(right):
        output.append(result(table, "recomputed_rows_equal", "exact equality", "row-count mismatch", False))
        return output
    equal = left.equals(right)
    details = ""
    if not equal:
        joined = left.merge(right, on=primary_key, how="outer", suffixes=("_live", "_rebuilt"), indicator=True)
        mismatch_rows: list[dict[str, Any]] = []
        for _, row in joined.iterrows():
            differences: dict[str, Any] = {}
            for column in compare_columns:
                if column in primary_key:
                    continue
                live_value = str(row.get(f"{column}_live", ""))
                rebuilt_value = str(row.get(f"{column}_rebuilt", ""))
                if live_value != rebuilt_value:
                    differences[column] = {"live": live_value, "rebuilt": rebuilt_value}
            if row["_merge"] != "both" or differences:
                mismatch_rows.append({
                    "key": {key: str(row.get(key, "")) for key in primary_key},
                    "merge": str(row["_merge"]),
                    "differences": differences,
                })
            if len(mismatch_rows) >= 20:
                break
        details = json.dumps(mismatch_rows, ensure_ascii=False)
    output.append(result(table, "recomputed_rows_equal", "exact equality excluding snapshot_date", "equal" if equal else "different", equal, details=details))
    return output


def rebuild_current_members(inputs: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    members = inputs["silver_members"]
    memberships = inputs["silver_member_memberships"]
    parties = inputs["silver_member_parties"]
    constituencies = inputs["silver_member_constituencies"]
    offices = inputs["silver_member_offices"]

    current_memberships = current_mod._select_current_or_latest(
        memberships,
        group_key="member_code",
        current_col="is_current",
        start_col="membership_start",
        end_col="membership_end",
    )
    current_parties = current_mod._select_current_or_latest(
        parties,
        group_key="member_code",
        current_col="is_current",
        start_col="party_start",
        end_col="party_end",
    )
    current_constituencies = current_mod._select_current_or_latest(
        constituencies,
        group_key="member_code",
        current_col="is_current",
        start_col="represent_start",
        end_col="represent_end",
    )
    current_offices = current_mod._aggregate_current_offices(offices)

    roster = members.copy()
    if not current_memberships.empty:
        roster = roster.merge(current_memberships[["member_code", "house_no", "membership_id"]], on="member_code", how="left", suffixes=("", "_membership"))
    else:
        roster["house_no"] = ""
        roster["membership_id"] = ""
    if not current_parties.empty:
        roster = roster.merge(current_parties[["member_code", "party_name"]], on="member_code", how="left", suffixes=("", "_party"))
    else:
        roster["party_name"] = ""
    if not current_constituencies.empty:
        roster = roster.merge(current_constituencies[["member_code", "constituency_name"]], on="member_code", how="left", suffixes=("", "_constituency"))
    else:
        roster["constituency_name"] = ""
    if not current_offices.empty:
        roster = roster.merge(current_offices[["member_code", "office_name"]], on="member_code", how="left")
    else:
        roster["office_name"] = ""

    roster["party_name"] = current_mod._coalesce_series(roster.get("party_name"), roster.get("latest_party_name"))
    roster["constituency_name"] = current_mod._coalesce_series(roster.get("constituency_name"), roster.get("latest_constituency_name"))
    roster["house_no"] = current_mod._coalesce_series(roster.get("house_no"), roster.get("latest_house_no"))
    roster["office_name"] = roster.get("office_name").fillna("") if "office_name" in roster else ""
    roster["snapshot_date"] = "validation"
    mask = current_mod._truthy(roster.get("is_current_member"))
    if mask.any():
        roster = roster[mask].copy()
    elif not current_memberships.empty:
        roster = roster[roster["member_code"].isin(set(current_memberships["member_code"].dropna().astype(str)))].copy()
    return current_mod._dedupe_rows(roster.reindex(columns=columns).copy(), primary_key="member_code")


def rebuild_yearly(current_members: pd.DataFrame, speeches: pd.DataFrame, votes: pd.DataFrame, divisions: pd.DataFrame, columns: list[str], primary_key: list[str]) -> pd.DataFrame:
    speech_metrics = yearly_mod._speech_metrics(speeches)
    vote_metrics = yearly_mod._vote_metrics(votes)
    division_counts = yearly_mod._division_counts(divisions, votes)
    grid = yearly_mod._member_year_grid(current_members, speech_metrics, vote_metrics, division_counts)
    metrics = grid.merge(speech_metrics, on=["member_code", "year"], how="left")
    metrics = metrics.merge(vote_metrics, on=["member_code", "year"], how="left")
    metrics = metrics.merge(division_counts, on="year", how="left")
    for column in ["speech_count", "debate_day_count", "votes_cast_count", "ta_count", "nil_count", "staon_count", "division_count"]:
        metrics[column] = pd.to_numeric(metrics.get(column), errors="coerce").fillna(0).astype(int)
    metrics["vote_participation_pct"] = metrics.apply(yearly_mod._participation_pct, axis=1)
    metrics["speech_rank"] = yearly_mod._rank_by_year(metrics, value_col="speech_count")
    metrics["vote_participation_rank"] = yearly_mod._rank_by_year(metrics, value_col="vote_participation_pct", tie_col="votes_cast_count")
    metrics["snapshot_date"] = "validation"
    return yearly_mod._dedupe_rows(metrics.reindex(columns=columns).copy(), primary_keys=primary_key)


def rebuild_monthly(current_members: pd.DataFrame, speeches: pd.DataFrame, votes: pd.DataFrame, columns: list[str], primary_key: list[str]) -> pd.DataFrame:
    speech_metrics = monthly_mod._speech_metrics(speeches)
    vote_metrics = monthly_mod._vote_metrics(votes)
    grid = monthly_mod._member_month_grid(current_members, speech_metrics, vote_metrics)
    metrics = grid.merge(speech_metrics, on=["member_code", "year_month"], how="left")
    metrics = metrics.merge(vote_metrics, on=["member_code", "year_month"], how="left")
    for column in ["speech_count", "debate_day_count", "votes_cast_count"]:
        metrics[column] = pd.to_numeric(metrics.get(column), errors="coerce").fillna(0).astype(int)
    metrics["snapshot_date"] = "validation"
    return monthly_mod._dedupe_rows(metrics.reindex(columns=columns).copy(), primary_keys=primary_key)


def rebuild_constituency(current_members: pd.DataFrame, speeches: pd.DataFrame, votes: pd.DataFrame, columns: list[str], primary_key: list[str]) -> pd.DataFrame:
    lookup = constituency_mod._member_constituency_lookup(current_members, votes)
    speech_metrics = constituency_mod._speech_metrics(speeches, lookup)
    vote_metrics = constituency_mod._vote_metrics(votes, lookup)
    member_counts = constituency_mod._member_counts(current_members, speech_metrics, vote_metrics)
    years = sorted(set(speech_metrics.get("year", pd.Series(dtype=str)).astype(str)) | set(vote_metrics.get("year", pd.Series(dtype=str)).astype(str)))
    constituencies = sorted(set(member_counts.get("constituency_name", pd.Series(dtype=str)).astype(str)) | set(speech_metrics.get("constituency_name", pd.Series(dtype=str)).astype(str)) | set(vote_metrics.get("constituency_name", pd.Series(dtype=str)).astype(str)))
    years = [value for value in years if value]
    constituencies = [value for value in constituencies if value]
    base = pd.DataFrame([{"constituency_name": constituency, "year": year} for year in years for constituency in constituencies])
    metrics = base.merge(member_counts, on="constituency_name", how="left")
    metrics = metrics.merge(speech_metrics, on=["constituency_name", "year"], how="left")
    metrics = metrics.merge(vote_metrics, on=["constituency_name", "year"], how="left")
    for column in ["member_count", "speech_count", "votes_cast_count"]:
        metrics[column] = pd.to_numeric(metrics.get(column), errors="coerce").fillna(0).astype(int)
    metrics["snapshot_date"] = "validation"
    return constituency_mod._dedupe_rows(metrics.reindex(columns=columns).copy(), primary_keys=primary_key)


def rebuild_facts(yearly: pd.DataFrame, monthly: pd.DataFrame, constituency: pd.DataFrame, current_members: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(fact_mod._member_yearly_facts(yearly, current_members, snapshot_date="validation"))
    rows.extend(fact_mod._member_monthly_facts(monthly, current_members, snapshot_date="validation"))
    rows.extend(fact_mod._constituency_yearly_facts(constituency, snapshot_date="validation"))
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["fact_id"], keep="first")
    return frame


def monthly_yearly_reconciliation(monthly: pd.DataFrame, yearly: pd.DataFrame) -> list[ValidationResult]:
    working = monthly.copy()
    working["year"] = working["year_month"].astype(str).str[:4]
    for column in ["speech_count", "votes_cast_count"]:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)
    sums = working.groupby(["member_code", "year"], as_index=False).agg(
        monthly_speech_count=("speech_count", "sum"),
        monthly_votes_cast_count=("votes_cast_count", "sum"),
    )
    expected = yearly[["member_code", "year", "speech_count", "votes_cast_count"]].copy()
    expected["speech_count"] = pd.to_numeric(expected["speech_count"], errors="coerce").fillna(0)
    expected["votes_cast_count"] = pd.to_numeric(expected["votes_cast_count"], errors="coerce").fillna(0)
    merged = expected.merge(sums, on=["member_code", "year"], how="outer").fillna(0)
    speech_bad = merged[merged["speech_count"] != merged["monthly_speech_count"]]
    vote_bad = merged[merged["votes_cast_count"] != merged["monthly_votes_cast_count"]]
    return [
        result("cross_table", "monthly_speech_sum_equals_yearly", 0, len(speech_bad), speech_bad.empty, details=speech_bad.head(20).to_json(orient="records")),
        result("cross_table", "monthly_vote_sum_equals_yearly", 0, len(vote_bad), vote_bad.empty, details=vote_bad.head(20).to_json(orient="records")),
    ]


def fact_source_reconciliation(facts: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> ValidationResult:
    missing: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for _, fact in facts.iterrows():
        source_table = str(fact["source_table"])
        source_key = str(fact["source_key"])
        metric_name = str(fact["metric_name"])
        source = sources.get(source_table)
        if source is None:
            missing.append({"source_table": source_table, "source_key": source_key})
            continue
        parts = source_key.split("|", 1)
        if source_table == "gold_member_activity_yearly":
            match = source[(source["member_code"].astype(str) == parts[0]) & (source["year"].astype(str) == parts[1])]
        elif source_table == "gold_member_activity_monthly":
            match = source[(source["member_code"].astype(str) == parts[0]) & (source["year_month"].astype(str) == parts[1])]
        elif source_table == "gold_constituency_activity_yearly":
            match = source[(source["constituency_name"].astype(str) == parts[0]) & (source["year"].astype(str) == parts[1])]
        else:
            match = pd.DataFrame()
        if match.empty or metric_name not in match.columns:
            missing.append({"source_table": source_table, "source_key": source_key, "metric": metric_name})
            continue
        actual = pd.to_numeric(pd.Series([fact["metric_value"]]), errors="coerce").iloc[0]
        expected = pd.to_numeric(pd.Series([match.iloc[0][metric_name]]), errors="coerce").iloc[0]
        if pd.isna(actual) or pd.isna(expected) or float(actual) != float(expected):
            mismatches.append({"fact_id": fact["fact_id"], "actual": str(actual), "expected": str(expected)})
    passed = not missing and not mismatches
    return result("gold_content_fact_pool", "every_fact_resolves_to_equal_source_metric", "0 missing and 0 mismatched", {"missing": len(missing), "mismatched": len(mismatches)}, passed, details=json.dumps({"missing_samples": missing[:20], "mismatch_samples": mismatches[:20]}, ensure_ascii=False))


def write_report(results: list[ValidationResult], live: dict[str, pd.DataFrame], rebuilt: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in results]
    pd.DataFrame(rows).to_csv(output_dir / "checkpoint5_results.csv", index=False)
    (output_dir / "checkpoint5_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary: list[dict[str, Any]] = []
    for table in GOLD_TABLES + ["cross_table"]:
        subset = [item for item in results if item.table == table]
        if not subset:
            continue
        summary.append({
            "table": table,
            "live_rows": len(live.get(table, pd.DataFrame())),
            "rebuilt_rows": len(rebuilt.get(table, pd.DataFrame())),
            "tests": len(subset),
            "passed": sum(item.status == "pass" for item in subset),
            "failed": sum(item.status == "fail" for item in subset),
        })
    (output_dir / "checkpoint5_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Oireachtas validation — Checkpoint 5", "", "| Table | Live rows | Rebuilt rows | Tests | Passed | Failed |", "|---|---:|---:|---:|---:|---:|"]
    for item in summary:
        lines.append(f"| {item['table']} | {item['live_rows']} | {item['rebuilt_rows']} | {item['tests']} | {item['passed']} | {item['failed']} |")
    failures = [item for item in results if item.status == "fail"]
    lines.extend(["", "## Findings", ""])
    if not failures:
        lines.append("No failures were found.")
    else:
        lines.extend(["| Table | Test | Expected | Actual | Details |", "|---|---|---|---|---|"])
        for item in failures:
            details = item.details.replace("|", "\\|").replace("\n", " ")[:1200]
            lines.append(f"| {item.table} | {item.test_name} | {item.expected_result} | {item.actual_result} | {details} |")
    (output_dir / "checkpoint5_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate Oireachtas gold tables by independent recomputation from live source tables.")
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--expectations", type=Path, default=Path("configs/oireachtas/validation_expectations.yml"))
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/checkpoint5"))
    p.add_argument("--fail-on-findings", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    expectations = load_expectations(args.expectations)
    registry = load_table_registry()
    prefix = str(expectations.get("logical_prefix") or "processed/oireachtas_unified/latest/csv")
    rules = expectations.get("table_rules") or {}
    s3 = boto3.client("s3", region_name=args.region)

    needed = [
        "silver_members", "silver_member_memberships", "silver_member_parties", "silver_member_constituencies", "silver_member_offices",
        "silver_speeches", "silver_member_votes", "silver_divisions", *GOLD_TABLES,
    ]
    live: dict[str, pd.DataFrame] = {}
    results: list[ValidationResult] = []
    for table in needed:
        csv_df, parquet_df = read_live(s3, bucket=args.bucket, table=table, prefix=prefix)
        live[table] = csv_df
        if table in GOLD_TABLES:
            results.extend(validate_dataframe(table=table, df=csv_df, expected_columns=registry[table].columns, primary_key=registry[table].primary_key, rules=rules.get(table) or {}, historical_start=str(expectations.get("historical_start") or "2024-11-29")))
            results.extend(validate_csv_parquet_equivalence(table=table, csv_df=csv_df, parquet_df=parquet_df))

    rebuilt: dict[str, pd.DataFrame] = {}
    rebuilt["gold_current_members"] = rebuild_current_members(live, registry["gold_current_members"].columns)
    rebuilt["gold_member_activity_yearly"] = rebuild_yearly(rebuilt["gold_current_members"], live["silver_speeches"], live["silver_member_votes"], live["silver_divisions"], registry["gold_member_activity_yearly"].columns, registry["gold_member_activity_yearly"].primary_key)
    rebuilt["gold_member_activity_monthly"] = rebuild_monthly(rebuilt["gold_current_members"], live["silver_speeches"], live["silver_member_votes"], registry["gold_member_activity_monthly"].columns, registry["gold_member_activity_monthly"].primary_key)
    rebuilt["gold_constituency_activity_yearly"] = rebuild_constituency(rebuilt["gold_current_members"], live["silver_speeches"], live["silver_member_votes"], registry["gold_constituency_activity_yearly"].columns, registry["gold_constituency_activity_yearly"].primary_key)
    rebuilt["gold_content_fact_pool"] = rebuild_facts(rebuilt["gold_member_activity_yearly"], rebuilt["gold_member_activity_monthly"], rebuilt["gold_constituency_activity_yearly"], rebuilt["gold_current_members"], registry["gold_content_fact_pool"].columns)

    for table in GOLD_TABLES:
        results.extend(compare_recomputed(table, live[table], rebuilt[table], registry[table].columns, registry[table].primary_key))
    results.extend(monthly_yearly_reconciliation(live["gold_member_activity_monthly"], live["gold_member_activity_yearly"]))
    results.append(fact_source_reconciliation(live["gold_content_fact_pool"], {
        "gold_member_activity_yearly": live["gold_member_activity_yearly"],
        "gold_member_activity_monthly": live["gold_member_activity_monthly"],
        "gold_constituency_activity_yearly": live["gold_constituency_activity_yearly"],
    }))

    write_report(results, {key: live[key] for key in GOLD_TABLES}, rebuilt, args.output_dir)
    failed = [item for item in results if item.status == "fail"]
    print(json.dumps({"tables": len(GOLD_TABLES), "tests": len(results), "passed": len(results) - len(failed), "failed": len(failed)}, indent=2))
    return 1 if args.fail_on_findings and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
