from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

import boto3
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas import table_bill_debates as bill_debates
from extract.oireachtas import table_bill_events as bill_events
from extract.oireachtas import table_bill_related_docs as bill_related_docs
from extract.oireachtas import table_bill_sponsors as bill_sponsors
from extract.oireachtas import table_bill_stages as bill_stages
from extract.oireachtas import table_bill_versions as bill_versions
from extract.oireachtas import table_bills as bills
from extract.oireachtas.client import OireachtasClient
from extract.oireachtas.io_s3 import get_bytes
from extract.oireachtas.schemas import load_table_registry
from process.oireachtas_validate_external import deterministic_sample, normalize_text
from process.oireachtas_validate_table import (
    ValidationResult,
    load_expectations,
    validate_csv_parquet_equivalence,
    validate_dataframe,
)

TABLES = [
    "silver_bills",
    "silver_bill_versions",
    "silver_bill_stages",
    "silver_bill_related_docs",
    "silver_bill_sponsors",
    "silver_bill_debates",
    "silver_bill_events",
]

PRIMARY_KEYS = {
    "silver_bills": "bill_id",
    "silver_bill_versions": "bill_version_id",
    "silver_bill_stages": "bill_stage_id",
    "silver_bill_related_docs": "related_doc_id",
    "silver_bill_sponsors": "bill_sponsor_id",
    "silver_bill_debates": "bill_debate_id",
    "silver_bill_events": "bill_event_id",
}

NORMALIZERS: dict[str, Callable[[Mapping[str, Any]], list[dict[str, Any]]]] = {
    "silver_bills": lambda item: [bills._normalise_bill_row(item, snapshot_date=date.today().isoformat(), endpoint="/legislation")],
    "silver_bill_versions": lambda item: bill_versions._normalise_version_rows(item, snapshot_date=date.today().isoformat()),
    "silver_bill_stages": lambda item: bill_stages._normalise_stage_rows(item, snapshot_date=date.today().isoformat()),
    "silver_bill_related_docs": lambda item: bill_related_docs._normalise_related_doc_rows(item, snapshot_date=date.today().isoformat()),
    "silver_bill_sponsors": lambda item: bill_sponsors._normalise_sponsor_rows(item, snapshot_date=date.today().isoformat()),
    "silver_bill_debates": lambda item: bill_debates._normalise_debate_rows(item, snapshot_date=date.today().isoformat()),
    "silver_bill_events": lambda item: bill_events._normalise_event_rows(item, snapshot_date=date.today().isoformat()),
}

COMPARE_FIELDS = {
    "silver_bills": [
        "bill_uri", "bill_no", "bill_year", "title", "short_title", "origin_house_uri",
        "origin_house_name", "bill_type", "status", "introduced_date", "last_event_date",
    ],
    "silver_bill_versions": [
        "bill_id", "version_label", "version_date", "format_pdf_uri", "format_pdf_url",
        "format_xml_uri", "format_xml_url", "source_file_id_pdf", "source_file_id_xml",
    ],
    "silver_bill_stages": [
        "bill_id", "stage_name", "stage_date", "house_uri", "house_name", "stage_outcome", "order_in_bill",
    ],
    "silver_bill_related_docs": [
        "bill_id", "related_doc_label", "related_doc_date", "doc_type", "language",
        "format_pdf_uri", "format_pdf_url", "format_xml_uri", "format_xml_url",
    ],
    "silver_bill_sponsors": [
        "bill_id", "sponsor_uri", "sponsor_name", "sponsor_role_uri", "sponsor_role_name", "is_primary", "sponsor_order",
    ],
    "silver_bill_debates": [
        "bill_id", "debate_id", "debate_uri", "debate_date", "debate_show_as",
        "debate_section_id", "chamber_uri", "chamber_name", "debate_order",
    ],
    "silver_bill_events": [
        "bill_id", "event_uri", "event_type_uri", "event_name", "event_date", "chamber_uri", "chamber_name", "event_order",
    ],
}

DATE_COLUMNS = {
    "silver_bill_versions": "version_date",
    "silver_bill_stages": "stage_date",
    "silver_bill_related_docs": "related_doc_date",
    "silver_bill_debates": "debate_date",
    "silver_bill_events": "event_date",
}

OFFICIAL_HOSTS = {"api.oireachtas.ie", "data.oireachtas.ie", "www.oireachtas.ie", "oireachtas.ie"}


def make_result(
    table: str,
    test: str,
    expected: Any,
    actual: Any,
    passed: bool,
    *,
    source: str = "",
    sample_id: str = "",
    details: str = "",
) -> ValidationResult:
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


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return normalize_text(value)


def read_live(s3: Any, *, bucket: str, table: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_key = f"{prefix}/{table}.csv"
    parquet_key = csv_key.replace("/csv/", "/parquet/").replace(".csv", ".parquet")
    csv_df = pd.read_csv(io.BytesIO(get_bytes(s3, bucket=bucket, key=csv_key)), dtype=str, keep_default_na=False)
    parquet_df = pd.read_parquet(io.BytesIO(get_bytes(s3, bucket=bucket, key=parquet_key))).astype(object)
    return csv_df, parquet_df


def dedupe(rows: list[dict[str, Any]], key: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if key not in frame.columns:
        return frame
    return frame.drop_duplicates(subset=[key], keep="first")


def fetch_official(
    client: OireachtasClient,
    *,
    date_start: str,
    date_end: str,
) -> tuple[dict[str, pd.DataFrame], str]:
    summary = client.get_json_summary(
        "/legislation",
        params={
            "chamber": "dail",
            "house_no": "34",
            "date_start": date_start,
            "date_end": date_end,
            "limit": 200,
        },
    )
    if not summary.ok or not summary.payload:
        raise RuntimeError(f"official legislation API failed: {summary.error or summary.status_code}")
    raw_results = [item for item in summary.payload.get("results", []) if isinstance(item, Mapping)]
    frames: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        rows: list[dict[str, Any]] = []
        for item in raw_results:
            rows.extend(NORMALIZERS[table](item))
        frames[table] = dedupe(rows, PRIMARY_KEYS[table])
    return frames, summary.url


def orphan_result(
    name: str,
    child: pd.DataFrame,
    child_column: str,
    parent: pd.DataFrame,
    parent_column: str,
    *,
    allow_blank: bool = False,
) -> ValidationResult:
    values = child[child_column].fillna("").astype(str)
    parent_values = set(parent[parent_column].fillna("").astype(str))
    mask = ~values.isin(parent_values)
    if allow_blank:
        mask &= values.str.strip() != ""
    invalid = child[mask]
    return make_result(
        "cross_table",
        name,
        0,
        len(invalid),
        invalid.empty,
        details=invalid.head(20).to_json(orient="records"),
    )


def unique_order_result(table: str, df: pd.DataFrame, order_column: str) -> ValidationResult:
    duplicates = df[df.duplicated(subset=["bill_id", order_column], keep=False)]
    return make_result(
        table,
        f"{order_column}_unique_per_bill",
        0,
        len(duplicates),
        duplicates.empty,
        details=duplicates.head(20).to_json(orient="records"),
    )


def lifecycle_result(bills_df: pd.DataFrame) -> ValidationResult:
    introduced = pd.to_datetime(bills_df["introduced_date"], errors="coerce")
    last_event = pd.to_datetime(bills_df["last_event_date"], errors="coerce")
    invalid = bills_df[introduced.notna() & last_event.notna() & (introduced > last_event)]
    return make_result(
        "silver_bills",
        "introduced_date_not_after_last_event_date",
        0,
        len(invalid),
        invalid.empty,
        details=invalid.head(20).to_json(orient="records"),
    )


def child_date_within_bill_result(
    table: str,
    child: pd.DataFrame,
    bills_df: pd.DataFrame,
    date_column: str,
) -> ValidationResult:
    merged = child[["bill_id", PRIMARY_KEYS[table], date_column]].merge(
        bills_df[["bill_id", "introduced_date", "last_event_date"]],
        on="bill_id",
        how="left",
    )
    child_date = pd.to_datetime(merged[date_column], errors="coerce")
    introduced = pd.to_datetime(merged["introduced_date"], errors="coerce")
    last_event = pd.to_datetime(merged["last_event_date"], errors="coerce")
    invalid = merged[
        child_date.notna()
        & (
            (introduced.notna() & (child_date < introduced))
            | (last_event.notna() & (child_date > last_event))
        )
    ]
    return make_result(
        table,
        f"{date_column}_within_bill_lifecycle",
        0,
        len(invalid),
        invalid.empty,
        details=invalid.head(20).to_json(orient="records"),
    )


def primary_sponsor_result(sponsors: pd.DataFrame) -> ValidationResult:
    working = sponsors.copy()
    working["__primary"] = working["is_primary"].fillna("").astype(str).str.lower().isin({"true", "1", "yes", "y"})
    counts = working[working["__primary"]].groupby("bill_id").size()
    invalid = counts[counts > 1]
    return make_result(
        "silver_bill_sponsors",
        "at_most_one_primary_sponsor_per_bill",
        0,
        len(invalid),
        invalid.empty,
        details=invalid.head(30).to_json(),
    )


def compare_official(
    *,
    table: str,
    live: pd.DataFrame,
    official: pd.DataFrame,
    source: str,
    sample_count: int,
) -> list[ValidationResult]:
    key = PRIMARY_KEYS[table]
    results: list[ValidationResult] = []
    if key not in official.columns:
        return [make_result(table, "official_primary_key_available", key, "missing", False, source=source)]
    live_keys = set(live[key].fillna("").astype(str))
    official_keys = set(official[key].fillna("").astype(str))
    shared = sorted((live_keys & official_keys) - {""})
    missing = sorted((official_keys - live_keys) - {""})
    extra = sorted((live_keys - official_keys) - {""})
    results.extend(
        [
            make_result(table, "official_key_overlap", "> 0", len(shared), bool(shared), source=source),
            make_result(
                table,
                "official_key_delta",
                "0 missing from live and 0 extra in live",
                {"missing_from_live": len(missing), "extra_in_live": len(extra)},
                not missing and not extra,
                source=source,
                details=json.dumps({"missing_samples": missing[:20], "extra_samples": extra[:20]}, ensure_ascii=False),
            ),
        ]
    )
    if not shared:
        return results
    sample = deterministic_sample(live[live[key].astype(str).isin(shared)], key_columns=[key], count=sample_count)
    official_index = official.set_index(key, drop=False)
    for _, live_row in sample.iterrows():
        sample_id = str(live_row[key])
        official_row = official_index.loc[sample_id]
        if isinstance(official_row, pd.DataFrame):
            official_row = official_row.iloc[0]
        mismatches: dict[str, dict[str, str]] = {}
        for field in COMPARE_FIELDS[table]:
            left = clean(live_row.get(field))
            right = clean(official_row.get(field))
            if left != right:
                mismatches[field] = {"live": left, "official": right}
        results.append(
            make_result(
                table,
                "official_api_sample_match",
                "all selected stable fields equal",
                "match" if not mismatches else "mismatch",
                not mismatches,
                source=source,
                sample_id=sample_id,
                details=json.dumps(mismatches, ensure_ascii=False, sort_keys=True),
            )
        )
    return results


def validate_document_links(
    versions_df: pd.DataFrame,
    related_df: pd.DataFrame,
    *,
    sample_count: int,
) -> list[ValidationResult]:
    records: list[dict[str, str]] = []
    for table, frame, key in [
        ("silver_bill_versions", versions_df, "bill_version_id"),
        ("silver_bill_related_docs", related_df, "related_doc_id"),
    ]:
        for _, row in frame.iterrows():
            for format_type in ["pdf", "xml"]:
                url = clean(row.get(f"format_{format_type}_url")) or clean(row.get(f"format_{format_type}_uri"))
                if url.startswith("http"):
                    records.append({"table": table, "record_id": clean(row.get(key)), "format_type": format_type, "url": url})
    frame = pd.DataFrame(records)
    if frame.empty:
        return [make_result("cross_table", "official_bill_document_links_available", "> 0", 0, False)]
    sample = deterministic_sample(frame, key_columns=["table", "record_id", "format_type"], count=sample_count)
    session = requests.Session()
    session.headers.update({"User-Agent": "eirepolitic-data-pipeline-validation/1.0"})
    results: list[ValidationResult] = []
    for _, row in sample.iterrows():
        table = str(row["table"])
        record_id = str(row["record_id"])
        format_type = str(row["format_type"])
        url = str(row["url"])
        host = (urlparse(url).hostname or "").lower()
        host_ok = host in OFFICIAL_HOSTS or host.endswith(".oireachtas.ie")
        results.append(make_result(table, "official_document_host", "official Oireachtas host", host, host_ok, source=url, sample_id=record_id))
        try:
            response = session.get(url, timeout=45, stream=True)
            status_ok = 200 <= response.status_code < 400
            content_type = response.headers.get("Content-Type", "").lower()
            type_ok = format_type in content_type or (format_type == "xml" and "text/plain" in content_type)
            results.append(make_result(table, "official_document_url_responds", "HTTP 2xx/3xx", response.status_code, status_ok, source=url, sample_id=record_id))
            results.append(make_result(table, "official_document_content_type", format_type, content_type, type_ok, source=url, sample_id=record_id))
            response.close()
        except Exception as exc:
            results.append(make_result(table, "official_document_url_responds", "successful response", f"{type(exc).__name__}: {exc}", False, source=url, sample_id=record_id))
    return results


def source_file_link_result(
    table: str,
    frame: pd.DataFrame,
    source_files: pd.DataFrame,
) -> ValidationResult:
    ids: list[str] = []
    for column in ["source_file_id_pdf", "source_file_id_xml"]:
        if column in frame.columns:
            ids.extend(value for value in frame[column].fillna("").astype(str) if value.strip())
    inventory = set(source_files["source_file_id"].fillna("").astype(str))
    missing = sorted(set(ids) - inventory)
    return make_result(
        table,
        "document_source_file_ids_resolve",
        0,
        len(missing),
        not missing,
        details=json.dumps({"missing_samples": missing[:30]}, ensure_ascii=False),
    )


def write_report(
    results: list[ValidationResult],
    live: dict[str, pd.DataFrame],
    official: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in results]
    pd.DataFrame(rows).to_csv(output_dir / "checkpoint4_results.csv", index=False)
    (output_dir / "checkpoint4_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary: list[dict[str, Any]] = []
    for table in TABLES + ["cross_table"]:
        subset = [item for item in results if item.table == table]
        if not subset:
            continue
        summary.append(
            {
                "table": table,
                "live_rows": len(live.get(table, pd.DataFrame())),
                "official_rows": len(official.get(table, pd.DataFrame())),
                "tests": len(subset),
                "passed": sum(item.status == "pass" for item in subset),
                "failed": sum(item.status == "fail" for item in subset),
            }
        )
    (output_dir / "checkpoint4_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Oireachtas validation — Checkpoint 4",
        "",
        "| Table | Live rows | Fresh official rows | Tests | Passed | Failed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(f"| {item['table']} | {item['live_rows']} | {item['official_rows']} | {item['tests']} | {item['passed']} | {item['failed']} |")
    failures = [item for item in results if item.status == "fail"]
    lines.extend(["", "## Findings", ""])
    if not failures:
        lines.append("No failures were found.")
    else:
        lines.extend(["| Table | Test | Expected | Actual | Sample | Details |", "|---|---|---|---|---|---|"])
        for item in failures:
            details = item.details.replace("|", "\\|").replace("\n", " ")[:1200]
            lines.append(f"| {item.table} | {item.test_name} | {item.expected_result} | {item.actual_result} | {item.sample_record_id} | {details} |")
    (output_dir / "checkpoint4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate Oireachtas legislation tables against live files and fresh official API data.")
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--expectations", type=Path, default=Path("configs/oireachtas/validation_expectations.yml"))
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/checkpoint4"))
    p.add_argument("--sample-count", type=int, default=5)
    p.add_argument("--fail-on-findings", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    expectations = load_expectations(args.expectations)
    registry = load_table_registry()
    prefix = str(expectations.get("logical_prefix") or "processed/oireachtas_unified/latest/csv")
    rules = expectations.get("table_rules") or {}
    historical_start = str(expectations.get("historical_start") or "2024-11-29")
    s3 = boto3.client("s3", region_name=args.region)

    live: dict[str, pd.DataFrame] = {}
    results: list[ValidationResult] = []
    for table in TABLES:
        csv_df, parquet_df = read_live(s3, bucket=args.bucket, table=table, prefix=prefix)
        live[table] = csv_df
        results.extend(
            validate_dataframe(
                table=table,
                df=csv_df,
                expected_columns=registry[table].columns,
                primary_key=registry[table].primary_key,
                rules=rules.get(table) or {},
                historical_start=historical_start,
            )
        )
        results.extend(validate_csv_parquet_equivalence(table=table, csv_df=csv_df, parquet_df=parquet_df))

    houses_df = read_live(s3, bucket=args.bucket, table="silver_houses", prefix=prefix)[0]
    debates_df = read_live(s3, bucket=args.bucket, table="silver_debate_records", prefix=prefix)[0]
    sections_df = read_live(s3, bucket=args.bucket, table="silver_debate_sections", prefix=prefix)[0]
    source_files_df = read_live(s3, bucket=args.bucket, table="silver_source_files", prefix=prefix)[0]

    for table in TABLES[1:]:
        results.append(orphan_result(f"{table}_bill", live[table], "bill_id", live["silver_bills"], "bill_id"))
    results.extend(
        [
            orphan_result("bill_origin_house", live["silver_bills"], "origin_house_uri", houses_df, "house_uri"),
            orphan_result("bill_stage_house", live["silver_bill_stages"], "house_uri", houses_df, "house_uri", allow_blank=True),
            orphan_result("bill_debate_debate", live["silver_bill_debates"], "debate_id", debates_df, "debate_id"),
            orphan_result("bill_debate_section", live["silver_bill_debates"], "debate_section_id", sections_df, "debate_section_id", allow_blank=True),
            orphan_result("bill_event_chamber", live["silver_bill_events"], "chamber_uri", houses_df, "house_uri", allow_blank=True),
        ]
    )
    results.extend(
        [
            lifecycle_result(live["silver_bills"]),
            unique_order_result("silver_bill_stages", live["silver_bill_stages"], "order_in_bill"),
            unique_order_result("silver_bill_sponsors", live["silver_bill_sponsors"], "sponsor_order"),
            unique_order_result("silver_bill_debates", live["silver_bill_debates"], "debate_order"),
            unique_order_result("silver_bill_events", live["silver_bill_events"], "event_order"),
            primary_sponsor_result(live["silver_bill_sponsors"]),
            source_file_link_result("silver_bill_versions", live["silver_bill_versions"], source_files_df),
            source_file_link_result("silver_bill_related_docs", live["silver_bill_related_docs"], source_files_df),
        ]
    )
    for table, date_column in DATE_COLUMNS.items():
        results.append(child_date_within_bill_result(table, live[table], live["silver_bills"], date_column))

    introduced = pd.to_datetime(live["silver_bills"]["introduced_date"], errors="coerce").dropna()
    latest = pd.to_datetime(live["silver_bills"]["last_event_date"], errors="coerce").dropna()
    date_start = introduced.min().date().isoformat()
    date_end = latest.max().date().isoformat()
    official, source = fetch_official(
        OireachtasClient(timeout_seconds=45, retries=5, backoff_seconds=2.0, sleep_seconds=0.1),
        date_start=date_start,
        date_end=date_end,
    )
    for table in TABLES:
        results.extend(
            compare_official(
                table=table,
                live=live[table],
                official=official[table],
                source=source,
                sample_count=max(1, args.sample_count),
            )
        )
    results.extend(validate_document_links(live["silver_bill_versions"], live["silver_bill_related_docs"], sample_count=max(1, args.sample_count)))

    write_report(results, live, official, args.output_dir)
    failed = [item for item in results if item.status == "fail"]
    print(
        json.dumps(
            {
                "tables": len(TABLES),
                "date_start": date_start,
                "date_end": date_end,
                "tests": len(results),
                "passed": len(results) - len(failed),
                "failed": len(failed),
            },
            indent=2,
        )
    )
    return 1 if args.fail_on_findings and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
