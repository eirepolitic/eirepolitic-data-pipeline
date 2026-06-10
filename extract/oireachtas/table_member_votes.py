"""Builder for the `silver_member_votes` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import parse_iso_date, stable_hash, utc_now_iso
from .schemas import TableSchema
from .table_divisions import CANONICAL_ENDPOINT, FALLBACK_ENDPOINT, _fetch_divisions, _record

TABLE_NAME = "silver_member_votes"

VOTE_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "taVotes": ("ta", "yes"),
    "nilVotes": ("nil", "no"),
    "staonVotes": ("staon", "abstain"),
}


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_member_votes(
    *,
    client: OireachtasClient,
    s3: Any,
    bucket: str,
    schema: TableSchema,
    limit: int,
    mode: str,
    chamber: str | None = None,
    house_no: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> TableBuildResult:
    """Fetch divisions and emit one row per member vote."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    date_start = date_start or "2025-01-01"
    date_end = date_end or "2025-01-31"
    params: dict[str, Any] = {
        "chamber_id": f"/ie/oireachtas/house/{chamber or 'dail'}/{house_no or '34'}",
        "date_start": date_start,
        "date_end": date_end,
        "limit": max(1, min(limit, 200)),
    }

    summary, endpoint_used, fallback_used = _fetch_divisions(client, params)
    if not summary.ok or not summary.payload:
        raise RuntimeError(f"Failed to fetch divisions: {summary.error or summary.status_code}")

    payload = dict(summary.payload)
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError(f"Unexpected division results type: {type(results).__name__}")

    rows: list[dict[str, Any]] = []
    division_diagnostics: list[dict[str, Any]] = []
    expected_member_rows = 0

    for item in results:
        if not isinstance(item, Mapping):
            continue
        record = _record(item)
        division_id = _division_id(record)
        vote_id = _text(record.get("voteId")) or _text(record.get("divisionId"))
        division_date = (
            parse_iso_date(record.get("date"))
            or parse_iso_date(record.get("voteDate"))
            or parse_iso_date(record.get("divisionDate"))
            or parse_iso_date(item.get("contextDate"))
        )
        tallies = record.get("tallies") if isinstance(record.get("tallies"), Mapping) else {}
        division_rows: list[dict[str, Any]] = []
        category_diagnostics: list[dict[str, Any]] = []

        for source_key, tally_value in tallies.items():
            if not isinstance(tally_value, Mapping):
                continue
            vote_code, vote_label = VOTE_CATEGORY_MAP.get(
                str(source_key),
                (_generic_vote_code(str(source_key)), _generic_vote_label(str(source_key))),
            )
            members = tally_value.get("members")
            member_items = members if isinstance(members, list) else []
            api_tally = _to_non_negative_int(tally_value.get("tally"))
            expected_member_rows += api_tally if api_tally is not None else len(member_items)

            category_output_count = 0
            for member_item in member_items:
                if not isinstance(member_item, Mapping):
                    continue
                member = member_item.get("member")
                if not isinstance(member, Mapping):
                    member = member_item
                member_code = _member_code(member)
                member_name = _first_text(member, "showAs", "fullName", "displayName", "name")
                party_name = _nested_name(member, "party", "partyName", "partyCode")
                constituency_name = _nested_name(
                    member,
                    "constituency",
                    "constituencyName",
                    "representName",
                )
                member_vote_id = f"member_vote:{stable_hash([division_id, member_code, vote_code], length=24)}"
                division_rows.append(
                    {
                        "member_vote_id": member_vote_id,
                        "division_id": division_id,
                        "vote_id": vote_id,
                        "division_date": division_date,
                        "member_code": member_code,
                        "member_name": member_name,
                        "vote_code": vote_code,
                        "vote_label": vote_label,
                        "party_name_at_vote": party_name,
                        "constituency_name_at_vote": constituency_name,
                        "snapshot_date": snapshot_date,
                        "_source_key": str(source_key),
                    }
                )
                category_output_count += 1

            category_diagnostics.append(
                {
                    "source_key": str(source_key),
                    "vote_code": vote_code,
                    "api_tally": api_tally,
                    "members_length": len(member_items),
                    "output_rows": category_output_count,
                }
            )

        rows.extend(division_rows)
        duplicate_member_codes = _duplicate_member_codes(division_rows)
        division_diagnostics.append(
            {
                "division_id": division_id,
                "vote_id": vote_id,
                "division_date": division_date,
                "output_rows": len(division_rows),
                "duplicate_member_codes": duplicate_member_codes,
                "categories": category_diagnostics,
            }
        )

    rows = _dedupe_rows(rows, primary_key="member_vote_id")
    diagnostic_df = pd.DataFrame(rows)
    df = diagnostic_df.reindex(columns=schema.columns)

    raw_key = f"raw/oireachtas_unified/api/divisions/snapshot_date={snapshot_date}/run_id={run_id}/page-00000.json"
    csv_key = f"processed/oireachtas_unified/silver_csv/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/{TABLE_NAME}.csv"
    parquet_key = f"processed/oireachtas_unified/silver/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/part-00000.parquet"
    latest_csv_key = f"processed/oireachtas_unified/latest/csv/{TABLE_NAME}.csv"
    latest_parquet_key = f"processed/oireachtas_unified/latest/parquet/{TABLE_NAME}.parquet"
    manifest_key = f"processed/oireachtas_unified/manifests/{TABLE_NAME}/run_id={run_id}.json"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"

    dq = _dq_results(df, schema, expected_member_rows=expected_member_rows)
    write_errors: list[str] = []
    schema_payload = {
        "table": TABLE_NAME,
        "primary_key": schema.primary_key,
        "columns": schema.columns,
        "row_count": int(len(df)),
    }
    s3_keys = {
        "raw_json": raw_key,
        "csv": csv_key,
        "parquet": parquet_key,
        "latest_csv": latest_csv_key,
        "latest_parquet": latest_parquet_key,
        "manifest": manifest_key,
        "review_sample": review_sample_key,
        "review_schema": review_schema_key,
        "review_manifest": review_manifest_key,
    }
    manifest = {
        "table": TABLE_NAME,
        "mode": mode,
        "status": "success" if dq["dq_status"] != "fail" else "failed",
        "run_id": run_id,
        "snapshot_date": snapshot_date,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "canonical_endpoint": CANONICAL_ENDPOINT,
        "fallback_endpoint": FALLBACK_ENDPOINT,
        "endpoint_used": endpoint_used,
        "fallback_used": fallback_used,
        "params": dict(summary.params),
        "url": summary.url,
        "status_code": summary.status_code,
        "raw_rows": len(results),
        "output_rows": int(len(df)),
        "expected_member_rows_from_tallies": expected_member_rows,
        "division_count": int(df["division_id"].nunique()) if not df.empty else 0,
        "member_count": int(df["member_code"].nunique()) if not df.empty else 0,
        "vote_codes": sorted(df["vote_code"].dropna().astype(str).unique().tolist()) if not df.empty else [],
        "party_name_at_vote_rows": int(df["party_name_at_vote"].notna().sum()) if not df.empty else 0,
        "constituency_name_at_vote_rows": int(df["constituency_name_at_vote"].notna().sum()) if not df.empty else 0,
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "division_diagnostics": division_diagnostics,
        "write_errors": write_errors,
        "s3_keys": s3_keys,
    }

    try:
        put_json(s3, bucket=bucket, key=raw_key, payload=payload)
        put_dataframe_csv(s3, bucket=bucket, key=csv_key, df=df)
        if not df.empty:
            put_dataframe_parquet(s3, bucket=bucket, key=parquet_key, df=df)
        put_dataframe_csv(s3, bucket=bucket, key=latest_csv_key, df=df)
        if not df.empty:
            put_dataframe_parquet(s3, bucket=bucket, key=latest_parquet_key, df=df)
        put_json(s3, bucket=bucket, key=manifest_key, payload=manifest)
        sample_df = df.head(10)
        put_dataframe_csv(s3, bucket=bucket, key=review_sample_key, df=sample_df)
        put_json(s3, bucket=bucket, key=review_schema_key, payload=schema_payload)
        put_json(s3, bucket=bucket, key=review_manifest_key, payload=manifest)
    except Exception as exc:
        write_errors.append(f"{type(exc).__name__}: {exc}")
        dq["dq_status"] = "fail"
        dq["checks"].append(
            {"check_name": "s3_write", "status": "fail", "message": write_errors[-1]}
        )
        manifest["status"] = "failed"
        manifest["dq_status"] = "fail"
        manifest["write_errors"] = write_errors

    sample_df = df.head(10)
    return TableBuildResult(
        table=TABLE_NAME,
        rows=sample_df.to_dict(orient="records"),
        manifest=manifest,
        schema=schema_payload,
        dq=dq,
        s3_keys=s3_keys,
    )


def _division_id(record: Mapping[str, Any]) -> str:
    for key in ("uri", "divisionUri", "voteUri", "voteId", "divisionId", "id", "eId"):
        value = _text(record.get(key))
        if value:
            return value
    return f"generated:division:{stable_hash(record, length=24)}"


def _member_code(member: Mapping[str, Any]) -> str | None:
    value = _first_text(member, "memberCode", "code", "id")
    if value:
        return value
    uri = _first_text(member, "uri", "memberUri")
    if uri and "/member/id/" in uri:
        return uri.split("/member/id/", 1)[1].split("/", 1)[0].strip() or None
    return None


def _nested_name(member: Mapping[str, Any], mapping_key: str, *scalar_keys: str) -> str | None:
    nested = member.get(mapping_key)
    if isinstance(nested, Mapping):
        value = _first_text(nested, "showAs", "name", "title")
        if value:
            return value
    return _first_text(member, *scalar_keys)


def _duplicate_member_codes(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        code = _text(row.get("member_code"))
        if not code:
            continue
        if code in seen:
            duplicates.add(code)
        seen.add(code)
    return sorted(duplicates)


def _generic_vote_code(source_key: str) -> str:
    value = source_key.strip()
    if value.lower().endswith("votes"):
        value = value[:-5]
    output: list[str] = []
    for char in value:
        if char.isupper() and output:
            output.append("_")
        output.append(char.lower())
    return "".join(output).strip("_") or "unknown"


def _generic_vote_label(source_key: str) -> str:
    return _generic_vote_code(source_key).replace("_", " ")


def _to_non_negative_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _text(mapping.get(key))
        if value:
            return value
    return None


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text or None


def _dedupe_rows(rows: list[dict[str, Any]], *, primary_key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get(primary_key) or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _dq_results(
    df: pd.DataFrame,
    schema: TableSchema,
    *,
    expected_member_rows: int,
) -> dict[str, Any]:
    pk = schema.primary_key[0]
    missing_columns = sorted(set(schema.columns) - set(df.columns))
    row_count = int(len(df))

    if row_count == 0 or pk not in df.columns:
        non_null_pk = unique_pk = division_ok = vote_id_ok = date_ok = member_ok = name_ok = code_ok = label_ok = row_count_ok = one_vote_ok = False
        division_count = 0
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        division_ok = bool(df["division_id"].notna().all() and (df["division_id"].astype(str).str.strip() != "").all())
        vote_id_ok = bool(df["vote_id"].notna().all() and (df["vote_id"].astype(str).str.strip() != "").all())
        date_ok = bool(df["division_date"].notna().all() and (df["division_date"].astype(str).str.strip() != "").all())
        member_ok = bool(df["member_code"].notna().all() and (df["member_code"].astype(str).str.strip() != "").all())
        name_ok = bool(df["member_name"].notna().all() and (df["member_name"].astype(str).str.strip() != "").all())
        code_ok = bool(df["vote_code"].notna().all() and (df["vote_code"].astype(str).str.strip() != "").all())
        label_ok = bool(df["vote_label"].notna().all() and (df["vote_label"].astype(str).str.strip() != "").all())
        row_count_ok = row_count == expected_member_rows
        one_vote_ok = bool(not df.duplicated(subset=["division_id", "member_code"]).any())
        division_count = int(df["division_id"].nunique())

    status = "pass" if all(
        [
            row_count > 0,
            not missing_columns,
            non_null_pk,
            unique_pk,
            division_ok,
            vote_id_ok,
            date_ok,
            member_ok,
            name_ok,
            code_ok,
            label_ok,
            row_count_ok,
            one_vote_ok,
        ]
    ) else "fail"

    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "expected_member_rows": expected_member_rows,
        "division_count": division_count,
        "primary_key": schema.primary_key,
        "primary_key_unique": unique_pk,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if non_null_pk else "fail"},
            {"check_name": "primary_key_unique", "status": "pass" if unique_pk else "fail"},
            {"check_name": "division_id_populated", "status": "pass" if division_ok else "fail"},
            {"check_name": "vote_id_populated", "status": "pass" if vote_id_ok else "fail"},
            {"check_name": "division_date_populated", "status": "pass" if date_ok else "fail"},
            {"check_name": "member_code_populated", "status": "pass" if member_ok else "fail"},
            {"check_name": "member_name_populated", "status": "pass" if name_ok else "fail"},
            {"check_name": "vote_code_populated", "status": "pass" if code_ok else "fail"},
            {"check_name": "vote_label_populated", "status": "pass" if label_ok else "fail"},
            {"check_name": "row_count_matches_tallies", "status": "pass" if row_count_ok else "fail", "metric_value": row_count, "threshold": expected_member_rows},
            {"check_name": "one_vote_per_member_per_division", "status": "pass" if one_vote_ok else "fail"},
        ],
    }


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
