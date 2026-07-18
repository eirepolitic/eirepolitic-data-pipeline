from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import boto3
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas import table_debate_records as debate_records
from extract.oireachtas import table_debate_sections as debate_sections
from extract.oireachtas import table_division_tallies as division_tallies
from extract.oireachtas import table_divisions as divisions
from extract.oireachtas import table_questions as questions
from extract.oireachtas.client import OireachtasClient
from extract.oireachtas.io_s3 import get_bytes
from extract.oireachtas.partitioned_fetch import get_date_partitioned_json_summary
from extract.oireachtas.schemas import load_table_registry
from process.oireachtas_validate_external import deterministic_sample, normalize_text
from process.oireachtas_validate_table import (
    ValidationResult,
    load_expectations,
    validate_csv_parquet_equivalence,
    validate_dataframe,
)

TABLES = [
    "silver_source_files",
    "silver_debate_records",
    "silver_debate_sections",
    "silver_speeches",
    "silver_divisions",
    "silver_division_tallies",
    "silver_member_votes",
    "silver_questions",
]
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


def read_live(s3: Any, *, bucket: str, table: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_key = f"{prefix}/{table}.csv"
    parquet_key = csv_key.replace("/csv/", "/parquet/").replace(".csv", ".parquet")
    csv_df = pd.read_csv(io.BytesIO(get_bytes(s3, bucket=bucket, key=csv_key)), dtype=str, keep_default_na=False)
    parquet_df = pd.read_parquet(io.BytesIO(get_bytes(s3, bucket=bucket, key=parquet_key))).astype(object)
    return csv_df, parquet_df


def nonblank(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return normalize_text(value)


def dedupe(rows: Iterable[dict[str, Any]], key: str) -> pd.DataFrame:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        value = nonblank(row.get(key))
        if value and value not in seen:
            seen.add(value)
            output.append(row)
    return pd.DataFrame(output)


def orphan_result(
    name: str,
    child: pd.DataFrame,
    child_column: str,
    parent: pd.DataFrame,
    parent_column: str,
    *,
    allow_blank: bool = False,
) -> ValidationResult:
    parent_values = set(parent[parent_column].fillna("").astype(str))
    values = child[child_column].fillna("").astype(str)
    mask = ~values.isin(parent_values)
    if allow_blank:
        mask &= values.str.strip() != ""
    invalid = child[mask]
    return make_result("cross_table", name, 0, len(invalid), invalid.empty, details=invalid.head(10).to_json(orient="records"))


def duplicate_group_result(table: str, name: str, df: pd.DataFrame, columns: list[str]) -> ValidationResult:
    duplicates = df[df.duplicated(subset=columns, keep=False)]
    return make_result(table, name, 0, len(duplicates), duplicates.empty, details=duplicates.head(10).to_json(orient="records"))


def fetch_official(
    client: OireachtasClient,
    *,
    date_start: str,
    date_end: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], list[Mapping[str, Any]]]:
    debate_summary = client.get_json_summary(
        "/debates",
        params={
            "chamber_id": "/ie/oireachtas/house/dail/34",
            "lang": "en",
            "date_start": date_start,
            "date_end": date_end,
            "limit": 200,
        },
    )
    division_summary, _, _ = divisions._fetch_divisions(
        client,
        {
            "chamber_id": "/ie/oireachtas/house/dail/34",
            "date_start": date_start,
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
            "date_start": date_start,
            "date_end": date_end,
            "limit": 200,
        },
    )
    for name, summary in {
        "debates": debate_summary,
        "divisions": division_summary,
        "questions": question_summary,
    }.items():
        if not summary.ok or not summary.payload:
            raise RuntimeError(f"official API call failed for {name}: {summary.error or summary.status_code}")

    snapshot = date.today().isoformat()
    debate_items = [item for item in debate_summary.payload.get("results", []) if isinstance(item, Mapping)]
    division_items = [item for item in division_summary.payload.get("results", []) if isinstance(item, Mapping)]
    question_items = [item for item in question_summary.payload.get("results", []) if isinstance(item, Mapping)]

    debate_rows = [debate_records._normalise_debate_row(item, snapshot_date=snapshot) for item in debate_items]
    section_rows: list[dict[str, Any]] = []
    for item in debate_items:
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

    division_rows = [divisions._normalise_division_row(item, snapshot_date=snapshot) for item in division_items]
    tally_rows: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    for item in division_items:
        record = divisions._record(item)
        division_id = division_tallies._division_id(record)
        tallies = record.get("tallies") if isinstance(record.get("tallies"), Mapping) else {}
        tally_rows.extend(division_tallies._normalise_tallies(tallies, division_id=division_id, snapshot_date=snapshot))
        vote_id = nonblank(record.get("voteId")) or nonblank(record.get("divisionId"))
        normalised_division = divisions._normalise_division_row(item, snapshot_date=snapshot)
        for source_key, tally_value in tallies.items():
            if not isinstance(tally_value, Mapping):
                continue
            vote_code, vote_label = division_tallies.VOTE_CATEGORY_MAP.get(
                str(source_key),
                (division_tallies._generic_vote_code(str(source_key)), division_tallies._generic_vote_label(str(source_key))),
            )
            members = tally_value.get("members") if isinstance(tally_value.get("members"), list) else []
            for member_item in members:
                if not isinstance(member_item, Mapping):
                    continue
                member = member_item.get("member") if isinstance(member_item.get("member"), Mapping) else member_item
                member_code = nonblank(member.get("memberCode")) or nonblank(member.get("code")) or nonblank(member.get("id"))
                if not member_code:
                    uri = nonblank(member.get("uri")) or nonblank(member.get("memberUri"))
                    if "/member/id/" in uri:
                        member_code = uri.split("/member/id/", 1)[1].split("/", 1)[0]
                vote_rows.append(
                    {
                        "division_id": division_id,
                        "vote_id": vote_id,
                        "division_date": normalised_division.get("division_date"),
                        "member_code": member_code,
                        "member_name": nonblank(member.get("showAs")) or nonblank(member.get("fullName")) or nonblank(member.get("name")),
                        "vote_code": vote_code,
                        "vote_label": vote_label,
                    }
                )

    question_rows = [questions._normalise_question_row(item, snapshot_date=snapshot) for item in question_items]
    frames = {
        "silver_debate_records": dedupe(debate_rows, "debate_id"),
        "silver_debate_sections": dedupe(section_rows, "debate_section_id"),
        "silver_divisions": dedupe(division_rows, "division_id"),
        "silver_division_tallies": dedupe(tally_rows, "division_tally_id"),
        "silver_member_votes": pd.DataFrame(vote_rows),
        "silver_questions": dedupe(question_rows, "question_id"),
    }
    sources = {
        "silver_debate_records": debate_summary.url,
        "silver_debate_sections": debate_summary.url,
        "silver_divisions": division_summary.url,
        "silver_division_tallies": division_summary.url,
        "silver_member_votes": division_summary.url,
        "silver_questions": question_summary.url,
    }
    return frames, sources, debate_items


def compare_overlap(
    *,
    table: str,
    live: pd.DataFrame,
    official: pd.DataFrame,
    key: str,
    fields: list[str],
    source: str,
    sample_count: int,
) -> list[ValidationResult]:
    output: list[ValidationResult] = []
    if key not in official.columns:
        return [make_result(table, "official_key_available", key, "missing", False, source=source)]
    live_keys = set(live[key].fillna("").astype(str))
    official_keys = set(official[key].fillna("").astype(str))
    overlap = sorted((live_keys & official_keys) - {""})
    output.append(make_result(table, "official_api_key_overlap", "> 0", len(overlap), bool(overlap), source=source))
    if not overlap:
        return output
    candidates = deterministic_sample(live[live[key].astype(str).isin(overlap)], key_columns=[key], count=sample_count)
    official_index = official.set_index(key, drop=False)
    for _, live_row in candidates.iterrows():
        sample_id = str(live_row[key])
        official_row = official_index.loc[sample_id]
        if isinstance(official_row, pd.DataFrame):
            official_row = official_row.iloc[0]
        mismatches: dict[str, dict[str, str]] = {}
        for field in fields:
            left = nonblank(live_row.get(field))
            right = nonblank(official_row.get(field))
            if left != right:
                mismatches[field] = {"live": left, "official": right}
        output.append(
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
    return output


def validate_tally_reconciliation(tallies: pd.DataFrame, votes: pd.DataFrame) -> ValidationResult:
    expected = tallies.copy()
    expected["member_count"] = pd.to_numeric(expected["member_count"], errors="coerce").fillna(-1).astype(int)
    actual = votes.groupby(["division_id", "vote_code"], as_index=False).size().rename(columns={"size": "vote_rows"})
    merged = expected.merge(actual, on=["division_id", "vote_code"], how="outer")
    merged["member_count"] = merged["member_count"].fillna(-1).astype(int)
    merged["vote_rows"] = merged["vote_rows"].fillna(0).astype(int)
    invalid = merged[merged["member_count"] != merged["vote_rows"]]
    return make_result(
        "cross_table",
        "division_tallies_equal_member_vote_counts",
        0,
        len(invalid),
        invalid.empty,
        details=invalid.head(20).to_json(orient="records"),
    )


def validate_speech_fields(speeches: pd.DataFrame) -> list[ValidationResult]:
    text = speeches["speech_text"].fillna("").astype(str)
    calculated_words = text.map(lambda value: len(value.split()))
    calculated_chars = text.map(len)
    calculated_hash = text.map(lambda value: sha256(value.encode("utf-8")).hexdigest()[:24])
    words = pd.to_numeric(speeches["word_count"], errors="coerce")
    chars = pd.to_numeric(speeches["char_count"], errors="coerce")
    word_bad = speeches[words != calculated_words]
    char_bad = speeches[chars != calculated_chars]
    hash_bad = speeches[speeches["speech_text_hash"].fillna("").astype(str) != calculated_hash]
    return [
        make_result("silver_speeches", "word_count_recomputed", 0, len(word_bad), word_bad.empty, details=word_bad.head(5).to_json(orient="records")),
        make_result("silver_speeches", "char_count_recomputed", 0, len(char_bad), char_bad.empty, details=char_bad.head(5).to_json(orient="records")),
        make_result("silver_speeches", "text_hash_recomputed", 0, len(hash_bad), hash_bad.empty, details=hash_bad.head(5).to_json(orient="records")),
    ]


def validate_xml_samples(
    speeches: pd.DataFrame,
    debates: pd.DataFrame,
    sections: pd.DataFrame,
    *,
    sample_count: int,
) -> list[ValidationResult]:
    output: list[ValidationResult] = []
    http = requests.Session()
    http.headers.update({"User-Agent": "eirepolitic-data-pipeline-validation/1.0"})
    available = debates[debates["source_xml_url"].fillna("").astype(str).str.startswith("http")]
    samples = deterministic_sample(available, key_columns=["debate_id"], count=min(sample_count, 3))
    for _, debate in samples.iterrows():
        debate_id = str(debate["debate_id"])
        url = str(debate["source_xml_url"])
        try:
            response = http.get(url, timeout=45)
            response.raise_for_status()
            xml_text = normalize_text(response.text)
            debate_speeches = speeches[speeches["debate_id"].astype(str) == debate_id]
            speech_sample = deterministic_sample(debate_speeches, key_columns=["speech_id"], count=1)
            if not speech_sample.empty:
                row = speech_sample.iloc[0]
                needle = normalize_text(row["speech_text"])[:100]
                output.append(make_result("silver_speeches", "official_xml_contains_speech_text", "text prefix present", needle, bool(needle and needle in xml_text), source=url, sample_id=str(row["speech_id"])))
            debate_sections_df = sections[sections["debate_id"].astype(str) == debate_id]
            section_sample = deterministic_sample(debate_sections_df, key_columns=["debate_section_id"], count=1)
            if not section_sample.empty:
                row = section_sample.iloc[0]
                heading = nonblank(row.get("heading")) or nonblank(row.get("show_as"))
                output.append(make_result("silver_debate_sections", "official_xml_contains_section_heading", "heading present", heading, bool(heading and heading in xml_text), source=url, sample_id=str(row["debate_section_id"])))
        except Exception as exc:
            output.append(make_result("silver_speeches", "official_xml_download", "successful official XML response", f"{type(exc).__name__}: {exc}", False, source=url, sample_id=debate_id))
    return output


def validate_source_files(df: pd.DataFrame, *, sample_count: int) -> list[ValidationResult]:
    output: list[ValidationResult] = []
    locators = df.copy()
    locators["__url"] = locators["format_url"].fillna("").astype(str)
    locators.loc[locators["__url"].str.strip() == "", "__url"] = locators["format_uri"].fillna("").astype(str)
    locators = locators[locators["__url"].str.startswith("http")]
    samples = deterministic_sample(locators, key_columns=["source_file_id"], count=sample_count)
    http = requests.Session()
    http.headers.update({"User-Agent": "eirepolitic-data-pipeline-validation/1.0"})
    for _, row in samples.iterrows():
        url = str(row["__url"])
        host = (urlparse(url).hostname or "").lower()
        official_host = host in OFFICIAL_HOSTS or host.endswith(".oireachtas.ie")
        output.append(make_result("silver_source_files", "official_source_host", "official Oireachtas host", host, official_host, source=url, sample_id=str(row["source_file_id"])))
        try:
            response = http.get(url, timeout=45, stream=True)
            ok = 200 <= response.status_code < 400
            actual_type = response.headers.get("Content-Type", "")
            expected_type = nonblank(row.get("content_type"))
            type_ok = not expected_type or expected_type.split(";", 1)[0] in actual_type or actual_type.split(";", 1)[0] in expected_type
            output.append(make_result("silver_source_files", "official_source_url_responds", "HTTP 2xx/3xx", response.status_code, ok, source=url, sample_id=str(row["source_file_id"])))
            output.append(make_result("silver_source_files", "official_source_content_type", expected_type or "known type", actual_type, type_ok, source=url, sample_id=str(row["source_file_id"])))
            response.close()
        except Exception as exc:
            output.append(make_result("silver_source_files", "official_source_url_responds", "successful response", f"{type(exc).__name__}: {exc}", False, source=url, sample_id=str(row["source_file_id"])))
    return output


def write_report(results: list[ValidationResult], live: dict[str, pd.DataFrame], official: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in results]
    pd.DataFrame(rows).to_csv(output_dir / "checkpoint3_results.csv", index=False)
    (output_dir / "checkpoint3_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
    (output_dir / "checkpoint3_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Oireachtas validation — Checkpoint 3", "", "| Table | Live rows | Fresh official rows | Tests | Passed | Failed |", "|---|---:|---:|---:|---:|---:|"]
    for item in summary:
        lines.append(f"| {item['table']} | {item['live_rows']} | {item['official_rows']} | {item['tests']} | {item['passed']} | {item['failed']} |")
    failures = [item for item in results if item.status == "fail"]
    lines.extend(["", "## Findings", ""])
    if not failures:
        lines.append("No failures were found.")
    else:
        lines.extend(["| Table | Test | Expected | Actual | Sample | Details |", "|---|---|---|---|---|---|"])
        for item in failures:
            details = item.details.replace("|", "\\|").replace("\n", " ")[:1000]
            lines.append(f"| {item.table} | {item.test_name} | {item.expected_result} | {item.actual_result} | {item.sample_record_id} | {details} |")
    (output_dir / "checkpoint3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate Oireachtas proceedings, votes, questions, and source files.")
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--expectations", type=Path, default=Path("configs/oireachtas/validation_expectations.yml"))
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/checkpoint3"))
    p.add_argument("--sample-count", type=int, default=5)
    p.add_argument("--fail-on-findings", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    expectations = load_expectations(args.expectations)
    registry = load_table_registry()
    prefix = str(expectations.get("logical_prefix") or "processed/oireachtas_unified/latest/csv")
    rules = expectations.get("table_rules") or {}
    s3 = boto3.client("s3", region_name=args.region)
    live: dict[str, pd.DataFrame] = {}
    results: list[ValidationResult] = []
    for table in TABLES:
        csv_df, parquet_df = read_live(s3, bucket=args.bucket, table=table, prefix=prefix)
        live[table] = csv_df
        results.extend(validate_dataframe(table=table, df=csv_df, expected_columns=registry[table].columns, primary_key=registry[table].primary_key, rules=rules.get(table) or {}, historical_start=str(expectations.get("historical_start") or "2024-11-29")))
        results.extend(validate_csv_parquet_equivalence(table=table, csv_df=csv_df, parquet_df=parquet_df))

    results.extend([
        orphan_result("debate_section_debate", live["silver_debate_sections"], "debate_id", live["silver_debate_records"], "debate_id"),
        orphan_result("speech_debate", live["silver_speeches"], "debate_id", live["silver_debate_records"], "debate_id"),
        orphan_result("speech_section", live["silver_speeches"], "debate_section_id", live["silver_debate_sections"], "debate_section_id"),
        orphan_result("speech_member", live["silver_speeches"], "speaker_member_code", read_live(s3, bucket=args.bucket, table="silver_members", prefix=prefix)[0], "member_code", allow_blank=True),
        orphan_result("tally_division", live["silver_division_tallies"], "division_id", live["silver_divisions"], "division_id"),
        orphan_result("member_vote_division", live["silver_member_votes"], "division_id", live["silver_divisions"], "division_id"),
        orphan_result("member_vote_member", live["silver_member_votes"], "member_code", read_live(s3, bucket=args.bucket, table="silver_members", prefix=prefix)[0], "member_code"),
        orphan_result("question_member", live["silver_questions"], "asked_by_member_code", read_live(s3, bucket=args.bucket, table="silver_members", prefix=prefix)[0], "member_code", allow_blank=True),
    ])
    results.extend([
        duplicate_group_result("silver_debate_sections", "section_order_unique_per_debate", live["silver_debate_sections"], ["debate_id", "section_order"]),
        duplicate_group_result("silver_speeches", "speech_order_unique_per_debate", live["silver_speeches"], ["debate_id", "speech_order"]),
        duplicate_group_result("silver_member_votes", "one_vote_per_member_per_division", live["silver_member_votes"], ["division_id", "member_code"]),
        validate_tally_reconciliation(live["silver_division_tallies"], live["silver_member_votes"]),
    ])
    results.extend(validate_speech_fields(live["silver_speeches"]))

    date_columns = {
        "silver_debate_records": "debate_date",
        "silver_speeches": "debate_date",
        "silver_divisions": "division_date",
        "silver_member_votes": "division_date",
        "silver_questions": "question_date",
    }
    date_start = min(
        pd.to_datetime(live[table][column], errors="coerce").dropna().min()
        for table, column in date_columns.items()
    ).date().isoformat()
    date_end = max(
        pd.to_datetime(live[table][column], errors="coerce").dropna().max()
        for table, column in date_columns.items()
    ).date().isoformat()
    client = OireachtasClient(timeout_seconds=45, retries=5, backoff_seconds=2.0, sleep_seconds=0.1)
    official, sources, _ = fetch_official(client, date_start=date_start, date_end=date_end)
    compare_specs = {
        "silver_debate_records": ("debate_id", ["debate_date", "house_uri", "house_no", "house_code", "show_as", "source_xml_uri"]),
        "silver_debate_sections": ("debate_section_id", ["debate_id", "section_eid", "section_order", "heading", "show_as"]),
        "silver_divisions": ("division_id", ["vote_id", "division_date", "house_uri", "house_no", "subject", "outcome"]),
        "silver_division_tallies": ("division_tally_id", ["division_id", "vote_code", "vote_label", "member_count"]),
        "silver_questions": ("question_id", ["question_date", "question_no", "question_type", "question_text", "asked_by_member_code", "asked_by_name", "to_minister_or_department"]),
    }
    for table, (key, fields) in compare_specs.items():
        results.extend(compare_overlap(table=table, live=live[table], official=official[table], key=key, fields=fields, source=sources[table], sample_count=max(1, args.sample_count)))

    live_votes = live["silver_member_votes"].copy()
    official_votes = official["silver_member_votes"].copy()
    live_votes["external_key"] = live_votes[["division_id", "member_code", "vote_code"]].fillna("").astype(str).agg("|".join, axis=1)
    official_votes["external_key"] = official_votes[["division_id", "member_code", "vote_code"]].fillna("").astype(str).agg("|".join, axis=1)
    results.extend(compare_overlap(table="silver_member_votes", live=live_votes, official=official_votes, key="external_key", fields=["division_id", "division_date", "member_code", "member_name", "vote_code", "vote_label"], source=sources["silver_member_votes"], sample_count=max(1, args.sample_count)))
    results.extend(validate_xml_samples(live["silver_speeches"], live["silver_debate_records"], live["silver_debate_sections"], sample_count=max(1, args.sample_count)))
    results.extend(validate_source_files(live["silver_source_files"], sample_count=max(1, args.sample_count)))

    write_report(results, live, official, args.output_dir)
    failed = [item for item in results if item.status == "fail"]
    print(json.dumps({"tables": len(TABLES), "date_start": date_start, "date_end": date_end, "tests": len(results), "passed": len(results) - len(failed), "failed": len(failed)}, indent=2))
    return 1 if args.fail_on_findings and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
