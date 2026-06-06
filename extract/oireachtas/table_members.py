"""Builder for the `silver_members` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import is_current_range, normalize_name, parse_iso_date, stable_hash, stable_record_hash, utc_now_iso
from .schemas import TableSchema


TABLE_NAME = "silver_members"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_members(
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
    """Fetch `/members`, normalize, and write silver_members outputs."""
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
        rows.extend(
            _normalise_member_record(record, item, snapshot_date=snapshot_date, endpoint=endpoint)
            for record in _iter_member_records(item)
        )

    rows = _dedupe_rows(rows, primary_key="member_code")
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
    schema_payload = {"table": TABLE_NAME, "primary_key": schema.primary_key, "columns": schema.columns, "row_count": int(len(df))}
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

    emitted: list[Mapping[str, Any]] = []
    for key in ("member", "person", "memberDetails"):
        value = item.get(key)
        if isinstance(value, Mapping):
            emitted.append(value)

    for nested_key in ("members", "memberList"):
        nested = item.get(nested_key)
        if isinstance(nested, list):
            for entry in nested:
                if isinstance(entry, Mapping):
                    emitted.append(_unwrap_member(entry))

    if emitted:
        return emitted

    if any(key in item for key in ("memberCode", "fullName", "showAs", "uri")):
        return [item]

    return list(_recursive_members(item))


def _recursive_members(value: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        if key in {"member", "person", "memberDetails"} and isinstance(child, Mapping):
            yield child
        elif key in {"members", "memberList"} and isinstance(child, list):
            for entry in child:
                if isinstance(entry, Mapping):
                    yield _unwrap_member(entry)
        elif isinstance(child, Mapping):
            yield from _recursive_members(child)


def _unwrap_member(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("member", "person", "memberDetails"):
        value = entry.get(key)
        if isinstance(value, Mapping):
            return value
    return entry


def _normalise_member_record(record: Mapping[str, Any], wrapper: Mapping[str, Any], *, snapshot_date: str, endpoint: str) -> dict[str, Any]:
    full_name = _first_text(record, "fullName", "showAs", "name", "displayName")
    first_name = _first_text(record, "firstName", "forename", "givenName")
    last_name = _first_text(record, "lastName", "surname", "familyName")
    display_name = _first_text(record, "showAs", "displayName", "fullName", "name")
    member_code = _first_text(record, "memberCode", "code", "id")
    member_uri = _first_text(record, "uri", "memberUri")
    gender = _first_text(record, "gender", "sex")

    if not full_name:
        full_name = normalize_name(" ".join(part for part in (first_name, last_name) if part))
    if not display_name:
        display_name = full_name
    if not member_code:
        member_code = stable_hash([member_uri, full_name, first_name, last_name], length=16)

    membership_rows = list(_iter_memberships(record, wrapper))
    latest = _latest_membership_context(membership_rows)

    return {
        "member_code": member_code,
        "member_uri": member_uri,
        "full_name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "display_name": display_name,
        "gender": gender,
        "member_key": stable_hash([member_code, member_uri, full_name], length=16),
        "is_current_member": latest["is_current_member"],
        "latest_party_name": latest["latest_party_name"],
        "latest_constituency_name": latest["latest_constituency_name"],
        "latest_house_no": latest["latest_house_no"],
        "source_endpoint": endpoint,
        "snapshot_date": snapshot_date,
        "source_hash": stable_record_hash(record),
    }


def _iter_memberships(record: Mapping[str, Any], wrapper: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for parent in (record, wrapper):
        if not isinstance(parent, Mapping):
            continue
        for key in ("memberships", "membership", "memberMemberships"):
            value = parent.get(key)
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, Mapping):
                        yield _unwrap_membership(entry)
            elif isinstance(value, Mapping):
                yield _unwrap_membership(value)


def _unwrap_membership(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("membership", "memberMembership"):
        value = entry.get(key)
        if isinstance(value, Mapping):
            return value
    return entry


def _latest_membership_context(memberships: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not memberships:
        return {
            "is_current_member": None,
            "latest_party_name": None,
            "latest_constituency_name": None,
            "latest_house_no": None,
        }

    enriched = [(_membership_end(m), _membership_start(m), m) for m in memberships]
    current = [(end, start, m) for end, start, m in enriched if is_current_range(start, end)]
    selected = sorted(current or enriched, key=lambda row: (row[0] or "9999-12-31", row[1] or ""), reverse=True)[0][2]

    return {
        "is_current_member": any(is_current_range(_membership_start(m), _membership_end(m)) for m in memberships),
        "latest_party_name": _party_name(selected),
        "latest_constituency_name": _constituency_name(selected),
        "latest_house_no": _house_no(selected),
    }


def _membership_start(membership: Mapping[str, Any]) -> str | None:
    date_range = dict(membership.get("dateRange") or membership.get("date_range") or {})
    return parse_iso_date(date_range.get("start") or membership.get("membershipStart") or membership.get("startDate") or membership.get("dateStart"))


def _membership_end(membership: Mapping[str, Any]) -> str | None:
    date_range = dict(membership.get("dateRange") or membership.get("date_range") or {})
    return parse_iso_date(date_range.get("end") or membership.get("membershipEnd") or membership.get("endDate") or membership.get("dateEnd"))


def _party_name(membership: Mapping[str, Any]) -> str | None:
    party = _first_mapping(membership, "party", "partyDetails")
    return _first_text(party, "showAs", "partyName", "name") or _first_text(membership, "partyName", "party")


def _constituency_name(membership: Mapping[str, Any]) -> str | None:
    constituency = _first_mapping(membership, "constituency", "constituencyOrPanel", "represent")
    return _first_text(constituency, "showAs", "name", "constituencyName") or _first_text(membership, "constituencyName", "represent")


def _house_no(membership: Mapping[str, Any]) -> str | None:
    house = _first_mapping(membership, "house", "houseRecord")
    return _first_text(house, "houseNo") or _first_text(membership, "houseNo", "house_no")


def _first_mapping(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


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
    required_columns = set(schema.columns)
    missing_columns = sorted(required_columns - set(df.columns))
    row_count = int(len(df))
    if row_count == 0 or pk not in df.columns:
        non_null_pk = False
        unique_pk = False
        name_populated = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        name_populated = bool(df["full_name"].notna().any() and (df["full_name"].astype(str).str.strip() != "").any())
    status = "pass" if row_count > 0 and not missing_columns and non_null_pk and unique_pk and name_populated else "fail"

    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": schema.primary_key,
        "primary_key_unique": unique_pk,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count, "threshold": 1},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if non_null_pk else "fail", "primary_key": pk},
            {"check_name": "primary_key_unique", "status": "pass" if unique_pk else "fail", "primary_key": pk},
            {"check_name": "full_name_populated", "status": "pass" if name_populated else "fail"},
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{table}_{stamp}"
