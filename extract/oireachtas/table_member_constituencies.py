"""Builder for the `silver_member_constituencies` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import is_current_range, parse_iso_date, stable_hash, utc_now_iso
from .schemas import TableSchema


TABLE_NAME = "silver_member_constituencies"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_member_constituencies(
    *,
    client: OireachtasClient,
    s3: Any,
    bucket: str,
    schema: TableSchema,
    limit: int,
    mode: str,
    chamber: str | None = None,
    house_no: str | None = None,
) -> TableBuildResult:
    """Fetch `/members`, explode membership represents, and write outputs."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    endpoint = schema.endpoint or "/members"
    params: dict[str, Any] = {"limit": max(1, min(limit, 200))}
    if chamber:
        params["chamber"] = chamber
    if house_no:
        params["house_no"] = house_no

    summary = client.get_json_summary(endpoint, params=params)
    if not summary.ok or not summary.payload:
        raise RuntimeError(f"Failed to fetch {endpoint}: {summary.error or summary.status_code}")

    payload = dict(summary.payload)
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError(f"Unexpected /members results type: {type(results).__name__}")

    rows: list[dict[str, Any]] = []
    for item in results:
        for member in _iter_member_records(item):
            for membership in _iter_memberships(member, item):
                for represent in _iter_represents(membership):
                    rows.append(
                        _normalise_constituency_row(
                            member,
                            membership,
                            represent,
                            snapshot_date=snapshot_date,
                        )
                    )

    rows = _dedupe_rows(rows, primary_key="member_constituency_id")
    df = pd.DataFrame(rows, columns=schema.columns)

    raw_key = f"raw/oireachtas_unified/api/members/snapshot_date={snapshot_date}/run_id={run_id}/page-00000.json"
    csv_key = f"processed/oireachtas_unified/silver_csv/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/{TABLE_NAME}.csv"
    parquet_key = f"processed/oireachtas_unified/silver/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/part-00000.parquet"
    latest_csv_key = f"processed/oireachtas_unified/latest/csv/{TABLE_NAME}.csv"
    latest_parquet_key = f"processed/oireachtas_unified/latest/parquet/{TABLE_NAME}.parquet"
    manifest_key = f"processed/oireachtas_unified/manifests/{TABLE_NAME}/run_id={run_id}.json"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"

    dq = _dq_results(df, schema)
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
        "endpoint": endpoint,
        "params": dict(summary.params),
        "url": summary.url,
        "status_code": summary.status_code,
        "raw_rows": len(results),
        "output_rows": int(len(df)),
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "date_fallback": "represent dateRange, otherwise parent membership dateRange",
        "raw_result_sample": results[:2],
        "raw_result_key_paths": sorted(_key_paths(results[0], max_depth=6)) if results and isinstance(results[0], Mapping) else [],
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
        dq["checks"].append({"check_name": "s3_write", "status": "fail", "message": write_errors[-1]})
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


def _iter_member_records(item: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(item, Mapping):
        return []
    member = item.get("member")
    if isinstance(member, Mapping):
        return [member]
    if any(key in item for key in ("memberCode", "fullName", "showAs", "uri")):
        return [item]
    return []


def _iter_memberships(member: Mapping[str, Any], wrapper: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for parent in (member, wrapper):
        if not isinstance(parent, Mapping):
            continue
        value = parent.get("memberships") or parent.get("membership") or parent.get("memberMemberships")
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, Mapping):
                    yield _unwrap(entry, "membership", "memberMembership")
        elif isinstance(value, Mapping):
            yield _unwrap(value, "membership", "memberMembership")


def _iter_represents(membership: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    value = membership.get("represents") or membership.get("represent")
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, Mapping):
                yield _unwrap(entry, "represent", "constituencyOrPanel", "constituency")
    elif isinstance(value, Mapping):
        yield _unwrap(value, "represent", "constituencyOrPanel", "constituency")


def _unwrap(entry: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, Mapping):
            return value
    return entry


def _normalise_constituency_row(
    member: Mapping[str, Any],
    membership: Mapping[str, Any],
    represent: Mapping[str, Any],
    *,
    snapshot_date: str,
) -> dict[str, Any]:
    member_code = _first_text(member, "memberCode", "code", "id")
    membership_start = _date_start(membership)
    membership_end = _date_end(membership)
    membership_id = _first_text(membership, "uri", "membershipUri") or (
        f"generated:membership:{stable_hash([member_code, membership_start, membership_end])}"
    )

    constituency_uri = _first_text(represent, "uri", "constituencyUri")
    constituency_name = _first_text(represent, "showAs", "name", "constituencyName", "representName")
    represent_start = _date_start(represent) or membership_start
    represent_end = _date_end(represent) or membership_end

    if not constituency_uri:
        constituency_uri = f"generated:constituency:{stable_hash([constituency_name, membership_id])}"

    member_constituency_id = (
        f"generated:member_constituency:"
        f"{stable_hash([membership_id, member_code, constituency_uri, represent_start, represent_end])}"
    )

    return {
        "member_constituency_id": member_constituency_id,
        "membership_id": membership_id,
        "member_code": member_code,
        "constituency_uri": constituency_uri,
        "constituency_name": constituency_name,
        "represent_start": represent_start,
        "represent_end": represent_end,
        "is_current": is_current_range(represent_start, represent_end),
        "snapshot_date": snapshot_date,
    }


def _date_start(record: Mapping[str, Any]) -> str | None:
    date_range = dict(record.get("dateRange") or record.get("date_range") or {})
    member_date_range = dict(record.get("memberDateRange") or {})
    return parse_iso_date(
        date_range.get("start")
        or member_date_range.get("start")
        or record.get("startDate")
        or record.get("dateStart")
    )


def _date_end(record: Mapping[str, Any]) -> str | None:
    date_range = dict(record.get("dateRange") or record.get("date_range") or {})
    member_date_range = dict(record.get("memberDateRange") or {})
    return parse_iso_date(
        date_range.get("end")
        or member_date_range.get("end")
        or record.get("endDate")
        or record.get("dateEnd")
    )


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is None or isinstance(value, (dict, list)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


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


def _dq_results(df: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    pk = schema.primary_key[0]
    missing_columns = sorted(set(schema.columns) - set(df.columns))
    row_count = int(len(df))

    if row_count == 0 or pk not in df.columns:
        non_null_pk = unique_pk = membership_ok = member_ok = uri_ok = name_ok = start_ok = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        membership_ok = bool(df["membership_id"].notna().all() and (df["membership_id"].astype(str).str.strip() != "").all())
        member_ok = bool(df["member_code"].notna().all() and (df["member_code"].astype(str).str.strip() != "").all())
        uri_ok = bool(df["constituency_uri"].notna().all() and (df["constituency_uri"].astype(str).str.strip() != "").all())
        name_ok = bool(df["constituency_name"].notna().all() and (df["constituency_name"].astype(str).str.strip() != "").all())
        start_ok = bool(df["represent_start"].notna().any() and (df["represent_start"].astype(str).str.strip() != "").any())

    status = "pass" if all(
        [row_count > 0, not missing_columns, non_null_pk, unique_pk, membership_ok, member_ok, uri_ok, name_ok, start_ok]
    ) else "fail"

    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": schema.primary_key,
        "primary_key_unique": unique_pk,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if non_null_pk else "fail"},
            {"check_name": "primary_key_unique", "status": "pass" if unique_pk else "fail"},
            {"check_name": "membership_id_populated", "status": "pass" if membership_ok else "fail"},
            {"check_name": "member_code_populated", "status": "pass" if member_ok else "fail"},
            {"check_name": "constituency_uri_populated", "status": "pass" if uri_ok else "fail"},
            {"check_name": "constituency_name_populated", "status": "pass" if name_ok else "fail"},
            {"check_name": "represent_start_any_populated", "status": "pass" if start_ok else "fail"},
        ],
    }


def _key_paths(value: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 6) -> set[str]:
    if depth >= max_depth:
        return set()
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(child_path)
            paths.update(_key_paths(child, prefix=child_path, depth=depth + 1, max_depth=max_depth))
    elif isinstance(value, list):
        list_path = f"{prefix}[]" if prefix else "[]"
        paths.add(list_path)
        if value:
            paths.update(_key_paths(value[0], prefix=list_path, depth=depth + 1, max_depth=max_depth))
    return paths


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
