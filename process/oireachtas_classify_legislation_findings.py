from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import boto3
import pandas as pd

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


def read_csv(s3: Any, bucket: str, table: str) -> pd.DataFrame:
    key = f"processed/oireachtas_unified/latest/csv/{table}.csv"
    return pd.read_csv(io.BytesIO(get_bytes(s3, bucket=bucket, key=key)), dtype=str, keep_default_na=False)


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def chamber_key(value: Any) -> str:
    text = clean(value).lower().rstrip("/")
    return text.rsplit("/", 1)[-1] if text else ""


def dedupe(rows: list[dict[str, Any]], key: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or key not in frame.columns:
        return frame
    return frame.drop_duplicates(subset=[key], keep="first")


def fetch_official(client: OireachtasClient, date_start: str, date_end: str) -> tuple[dict[str, pd.DataFrame], str]:
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
    items = [item for item in summary.payload.get("results", []) if isinstance(item, Mapping)]
    snapshot = date.today().isoformat()
    normalizers = {
        "silver_bills": lambda item: [bills._normalise_bill_row(item, snapshot_date=snapshot, endpoint="/legislation")],
        "silver_bill_versions": lambda item: bill_versions._normalise_version_rows(item, snapshot_date=snapshot),
        "silver_bill_stages": lambda item: bill_stages._normalise_stage_rows(item, snapshot_date=snapshot),
        "silver_bill_related_docs": lambda item: bill_related_docs._normalise_related_doc_rows(item, snapshot_date=snapshot),
        "silver_bill_sponsors": lambda item: bill_sponsors._normalise_sponsor_rows(item, snapshot_date=snapshot),
        "silver_bill_debates": lambda item: bill_debates._normalise_debate_rows(item, snapshot_date=snapshot),
        "silver_bill_events": lambda item: bill_events._normalise_event_rows(item, snapshot_date=snapshot),
    }
    frames: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        rows: list[dict[str, Any]] = []
        for item in items:
            rows.extend(normalizers[table](item))
        frames[table] = dedupe(rows, PRIMARY_KEYS[table])
    return frames, summary.url


def comparable_scope(live: dict[str, pd.DataFrame], official: dict[str, pd.DataFrame]) -> dict[str, Any]:
    live_bill_ids = set(live["silver_bills"]["bill_id"].astype(str))
    result: dict[str, Any] = {}
    for table in TABLES:
        key = PRIMARY_KEYS[table]
        official_frame = official[table]
        if table == "silver_bills":
            scoped = official_frame[official_frame["bill_id"].astype(str).isin(live_bill_ids)]
        else:
            scoped = official_frame[official_frame["bill_id"].astype(str).isin(live_bill_ids)]
        live_keys = set(live[table][key].astype(str))
        official_keys = set(scoped[key].astype(str)) if key in scoped.columns else set()
        result[table] = {
            "live_rows": int(len(live[table])),
            "official_rows_for_live_bills": int(len(scoped)),
            "shared_keys": int(len(live_keys & official_keys)),
            "missing_from_live": int(len(official_keys - live_keys)),
            "extra_in_live": int(len(live_keys - official_keys)),
            "missing_samples": sorted(official_keys - live_keys)[:20],
            "extra_samples": sorted(live_keys - official_keys)[:20],
        }
    return result


def chamber_model(live: dict[str, pd.DataFrame], houses: pd.DataFrame) -> dict[str, Any]:
    known = set(houses["house_code"].astype(str).str.lower()) | set(houses["chamber"].astype(str).str.lower())
    known |= {"dail", "seanad", "oireachtas"}
    checks: dict[str, Any] = {}
    for table, column in [
        ("silver_bills", "origin_house_uri"),
        ("silver_bill_stages", "house_uri"),
        ("silver_bill_debates", "chamber_uri"),
        ("silver_bill_events", "chamber_uri"),
    ]:
        values = live[table][column].fillna("").astype(str)
        normalized = values.map(chamber_key)
        invalid = live[table][(values.str.strip() != "") & ~normalized.isin(known)]
        checks[table] = {
            "column": column,
            "known_chamber_keys": sorted(known),
            "invalid_rows": int(len(invalid)),
            "invalid_samples": invalid.head(20).to_dict(orient="records"),
            "direct_uri_join_is_valid": False,
            "classification": "definition URI must be normalized to chamber key before joining to house-instance URIs",
        }
    return checks


def debate_links(live: dict[str, pd.DataFrame], debates: pd.DataFrame, sections: pd.DataFrame) -> dict[str, Any]:
    links = live["silver_bill_debates"].copy()
    links["__chamber"] = links["chamber_uri"].map(chamber_key)
    links["__date"] = pd.to_datetime(links["debate_date"], errors="coerce")
    debate_dates = pd.to_datetime(debates["debate_date"], errors="coerce").dropna()
    min_date = debate_dates.min()
    max_date = debate_dates.max()
    comparable = links[(links["__chamber"] == "dail") & links["__date"].between(min_date, max_date, inclusive="both")]
    debate_ids = set(debates["debate_id"].astype(str))
    missing_debate = comparable[~comparable["debate_id"].astype(str).isin(debate_ids)]

    section_pairs = set(zip(sections["debate_id"].astype(str), sections["section_eid"].astype(str)))
    has_section = comparable["debate_section_id"].fillna("").astype(str).str.strip() != ""
    pair_values = list(zip(comparable["debate_id"].astype(str), comparable["debate_section_id"].astype(str)))
    pair_match = pd.Series([pair in section_pairs for pair in pair_values], index=comparable.index)
    missing_section = comparable[has_section & ~pair_match]

    return {
        "live_bill_debate_rows": int(len(links)),
        "comparable_current_dail_rows": int(len(comparable)),
        "out_of_scope_rows": int(len(links) - len(comparable)),
        "missing_comparable_debates": int(len(missing_debate)),
        "missing_comparable_sections_by_debate_and_eid": int(len(missing_section)),
        "missing_debate_samples": missing_debate.head(20).to_dict(orient="records"),
        "missing_section_samples": missing_section.head(20).to_dict(orient="records"),
        "classification": "bill debate links include Seanad and older proceedings; section links use local section_eid, not debate_section_id primary keys",
    }


def sponsor_model(live: dict[str, pd.DataFrame], official: dict[str, pd.DataFrame]) -> dict[str, Any]:
    live_sponsors = live["silver_bill_sponsors"].copy()
    official_sponsors = official["silver_bill_sponsors"].copy()
    for frame in (live_sponsors, official_sponsors):
        frame["__primary"] = frame["is_primary"].fillna("").astype(str).str.lower().isin({"true", "1", "yes", "y"})
    counts = live_sponsors[live_sponsors["__primary"]].groupby("bill_id").size()
    affected = sorted(counts[counts > 1].index.astype(str))
    details: list[dict[str, Any]] = []
    for bill_id in affected:
        live_rows = live_sponsors[(live_sponsors["bill_id"].astype(str) == bill_id) & live_sponsors["__primary"]]
        official_rows = official_sponsors[(official_sponsors["bill_id"].astype(str) == bill_id) & official_sponsors["__primary"]]
        details.append(
            {
                "bill_id": bill_id,
                "live_primary_sponsors": live_rows[["sponsor_name", "sponsor_uri", "sponsor_role_name", "sponsor_order"]].to_dict(orient="records"),
                "official_primary_sponsors": official_rows[["sponsor_name", "sponsor_uri", "sponsor_role_name", "sponsor_order"]].to_dict(orient="records"),
                "matches_official_count": int(len(live_rows)) == int(len(official_rows)),
            }
        )
    return {
        "bills_with_multiple_primary_sponsors": len(affected),
        "details": details,
        "classification": "multiple primary sponsors are allowed when the official source lists co-sponsors",
    }


def debate_identity(live: dict[str, pd.DataFrame], official: dict[str, pd.DataFrame]) -> dict[str, Any]:
    live_frame = live["silver_bill_debates"].copy()
    official_frame = official["silver_bill_debates"].copy()
    live_bill_ids = set(live["silver_bills"]["bill_id"].astype(str))
    official_frame = official_frame[official_frame["bill_id"].astype(str).isin(live_bill_ids)]
    business_columns = ["bill_id", "debate_id", "debate_section_id", "debate_show_as", "debate_date", "chamber_uri"]
    for frame in (live_frame, official_frame):
        frame["__business_key"] = frame[business_columns].fillna("").astype(str).agg("|".join, axis=1)
    live_id = set(live_frame["bill_debate_id"].astype(str))
    official_id = set(official_frame["bill_debate_id"].astype(str))
    live_business = set(live_frame["__business_key"].astype(str))
    official_business = set(official_frame["__business_key"].astype(str))
    live_only_ids = live_id - official_id
    official_only_ids = official_id - live_id
    live_only_business = live_business - official_business
    official_only_business = official_business - live_business
    return {
        "live_rows": int(len(live_frame)),
        "official_rows_for_live_bills": int(len(official_frame)),
        "id_missing_from_live": int(len(official_only_ids)),
        "id_extra_in_live": int(len(live_only_ids)),
        "business_rows_missing_from_live": int(len(official_only_business)),
        "business_rows_extra_in_live": int(len(live_only_business)),
        "id_drift_explained_by_same_business_row": int(max(0, min(len(live_only_ids), len(official_only_ids)) - max(len(live_only_business), len(official_only_business)))),
        "live_only_business_samples": sorted(live_only_business)[:20],
        "official_only_business_samples": sorted(official_only_business)[:20],
    }


def historical_scope(bills_frame: pd.DataFrame, historical_start: str) -> dict[str, Any]:
    introduced = pd.to_datetime(bills_frame["introduced_date"], errors="coerce")
    last_event = pd.to_datetime(bills_frame["last_event_date"], errors="coerce")
    start = pd.Timestamp(historical_start)
    older = bills_frame[introduced.notna() & (introduced < start)].copy()
    older_active = older[pd.to_datetime(older["last_event_date"], errors="coerce") >= start]
    older_inactive = older[pd.to_datetime(older["last_event_date"], errors="coerce") < start]
    return {
        "historical_start": historical_start,
        "bills_introduced_before_start": int(len(older)),
        "older_bills_with_event_on_or_after_start": int(len(older_active)),
        "older_bills_with_last_event_before_start": int(len(older_inactive)),
        "older_inactive_samples": older_inactive.head(20)[["bill_id", "title", "introduced_date", "last_event_date", "status"]].to_dict(orient="records"),
        "classification": "bill introduction date is not a valid lower-bound coverage test; older bills can remain active or be updated in the validation window",
    }


def write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "legislation_findings_classification.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Legislation findings classification", ""]
    lines.extend([
        "## Comparable official scope",
        "",
        "| Table | Live | Official for live bills | Shared | Missing from live | Extra in live |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for table, item in payload["comparable_scope"].items():
        lines.append(f"| {table} | {item['live_rows']} | {item['official_rows_for_live_bills']} | {item['shared_keys']} | {item['missing_from_live']} | {item['extra_in_live']} |")
    lines.extend(["", "## Sponsor classification", "", f"Bills with multiple primary sponsors: {payload['sponsor_model']['bills_with_multiple_primary_sponsors']}", ""])
    for item in payload["sponsor_model"]["details"]:
        lines.append(f"- `{item['bill_id']}`: live={json.dumps(item['live_primary_sponsors'], ensure_ascii=False)}; official={json.dumps(item['official_primary_sponsors'], ensure_ascii=False)}")
    lines.extend([
        "",
        "## Debate-link classification",
        "",
        f"- Comparable current-Dáil links: {payload['debate_links']['comparable_current_dail_rows']}",
        f"- Missing comparable debate records: {payload['debate_links']['missing_comparable_debates']}",
        f"- Missing comparable sections using `(debate_id, section_eid)`: {payload['debate_links']['missing_comparable_sections_by_debate_and_eid']}",
        f"- Out-of-scope Seanad/older links: {payload['debate_links']['out_of_scope_rows']}",
        "",
        "## Historical bill scope",
        "",
        f"- Bills introduced before {payload['historical_scope']['historical_start']}: {payload['historical_scope']['bills_introduced_before_start']}",
        f"- Older bills active/updated after start: {payload['historical_scope']['older_bills_with_event_on_or_after_start']}",
        f"- Older bills whose last event also predates start: {payload['historical_scope']['older_bills_with_last_event_before_start']}",
        "",
        "## Classifications",
        "",
        "- House URI failures are validator-model errors: legislation uses chamber-definition URIs, while `silver_houses` uses numbered house-instance URIs.",
        "- Bill section links must join on `(debate_id, section_eid)`.",
        "- Multiple primary sponsors are permitted when present in the official payload.",
        "- Raw official row-count differences are not comparable until restricted to the same live bill IDs.",
    ])
    (output_dir / "legislation_findings_classification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classify Checkpoint 4 legislation validation findings.")
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--historical-start", default="2024-11-29")
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/checkpoint4_classification"))
    return p


def main() -> int:
    args = parser().parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    live = {table: read_csv(s3, args.bucket, table) for table in TABLES}
    houses = read_csv(s3, args.bucket, "silver_houses")
    debates = read_csv(s3, args.bucket, "silver_debate_records")
    sections = read_csv(s3, args.bucket, "silver_debate_sections")
    introduced = pd.to_datetime(live["silver_bills"]["introduced_date"], errors="coerce").dropna()
    last_event = pd.to_datetime(live["silver_bills"]["last_event_date"], errors="coerce").dropna()
    official, source = fetch_official(
        OireachtasClient(timeout_seconds=45, retries=5, backoff_seconds=2.0, sleep_seconds=0.1),
        introduced.min().date().isoformat(),
        last_event.max().date().isoformat(),
    )
    payload = {
        "official_source": source,
        "comparable_scope": comparable_scope(live, official),
        "chamber_model": chamber_model(live, houses),
        "debate_links": debate_links(live, debates, sections),
        "sponsor_model": sponsor_model(live, official),
        "debate_identity": debate_identity(live, official),
        "historical_scope": historical_scope(live["silver_bills"], args.historical_start),
    }
    write_report(payload, args.output_dir)
    print(json.dumps({
        "comparable_scope": {table: {"missing": item["missing_from_live"], "extra": item["extra_in_live"]} for table, item in payload["comparable_scope"].items()},
        "missing_comparable_debates": payload["debate_links"]["missing_comparable_debates"],
        "missing_comparable_sections": payload["debate_links"]["missing_comparable_sections_by_debate_and_eid"],
        "multiple_primary_sponsor_bills": payload["sponsor_model"]["bills_with_multiple_primary_sponsors"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
