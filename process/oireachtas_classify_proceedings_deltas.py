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

from extract.oireachtas import table_debate_records as debate_records
from extract.oireachtas import table_debate_sections as debate_sections
from extract.oireachtas import table_questions as questions
from extract.oireachtas.client import OireachtasClient
from extract.oireachtas.io_s3 import get_bytes
from extract.oireachtas.partitioned_fetch import get_date_partitioned_json_summary


def read_csv(s3: Any, bucket: str, table: str) -> pd.DataFrame:
    key = f"processed/oireachtas_unified/latest/csv/{table}.csv"
    return pd.read_csv(io.BytesIO(get_bytes(s3, bucket=bucket, key=key)), dtype=str, keep_default_na=False)


def dedupe(rows: list[dict[str, Any]], key: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or key not in frame.columns:
        return frame
    return frame.drop_duplicates(subset=[key], keep="first")


def fetch_official(client: OireachtasClient, date_start: str, date_end: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
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
    if not debate_summary.ok or not debate_summary.payload:
        raise RuntimeError(f"debates API failed: {debate_summary.error or debate_summary.status_code}")
    if not question_summary.ok or not question_summary.payload:
        raise RuntimeError(f"questions API failed: {question_summary.error or question_summary.status_code}")

    snapshot = date.today().isoformat()
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
    return (
        dedupe(section_rows, "debate_section_id"),
        dedupe(question_rows, "question_id"),
        {"sections": debate_summary.url, "questions": question_summary.url},
    )


def compare_keys(live: pd.DataFrame, official: pd.DataFrame, key: str, date_column: str) -> dict[str, Any]:
    live_keys = set(live[key].astype(str))
    official_keys = set(official[key].astype(str))
    missing_live_keys = official_keys - live_keys
    extra_live_keys = live_keys - official_keys

    missing_live = official[official[key].astype(str).isin(missing_live_keys)].copy()
    extra_live = live[live[key].astype(str).isin(extra_live_keys)].copy()
    for frame in (missing_live, extra_live):
        if date_column in frame.columns:
            frame["__date"] = pd.to_datetime(frame[date_column], errors="coerce").dt.date.astype(str)

    def counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty or "__date" not in frame.columns:
            return []
        grouped = frame.groupby("__date", dropna=False).size().reset_index(name="count").sort_values(["count", "__date"], ascending=[False, True])
        return grouped.to_dict(orient="records")

    return {
        "live_rows": int(len(live)),
        "official_rows": int(len(official)),
        "shared_keys": int(len(live_keys & official_keys)),
        "missing_from_live": int(len(missing_live_keys)),
        "extra_in_live": int(len(extra_live_keys)),
        "missing_from_live_by_date": counts(missing_live),
        "extra_in_live_by_date": counts(extra_live),
        "missing_from_live_samples": missing_live.head(20).drop(columns=["__date"], errors="ignore").to_dict(orient="records"),
        "extra_in_live_samples": extra_live.head(20).drop(columns=["__date"], errors="ignore").to_dict(orient="records"),
    }


def write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "proceedings_delta_classification.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Proceedings row-count delta classification", ""]
    for name in ["debate_sections", "questions"]:
        item = payload[name]
        lines.extend([
            f"## {name}",
            "",
            f"- Live rows: {item['live_rows']}",
            f"- Fresh official rows: {item['official_rows']}",
            f"- Shared keys: {item['shared_keys']}",
            f"- Missing from live: {item['missing_from_live']}",
            f"- Extra in live: {item['extra_in_live']}",
            "",
            "### Missing from live by date",
            "",
        ])
        if item["missing_from_live_by_date"]:
            for row in item["missing_from_live_by_date"][:30]:
                lines.append(f"- {row['__date']}: {row['count']}")
        else:
            lines.append("None.")
        lines.extend(["", "### Extra in live by date", ""])
        if item["extra_in_live_by_date"]:
            for row in item["extra_in_live_by_date"][:30]:
                lines.append(f"- {row['__date']}: {row['count']}")
        else:
            lines.append("None.")
        lines.append("")
    (output_dir / "proceedings_delta_classification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classify live-versus-current API row-count deltas for proceedings tables.")
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/checkpoint3_delta"))
    return p


def main() -> int:
    args = parser().parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    live_sections = read_csv(s3, args.bucket, "silver_debate_sections")
    live_questions = read_csv(s3, args.bucket, "silver_questions")
    dates = pd.concat([
        pd.to_datetime(live_questions["question_date"], errors="coerce"),
        pd.to_datetime(read_csv(s3, args.bucket, "silver_debate_records")["debate_date"], errors="coerce"),
    ]).dropna()
    date_start = dates.min().date().isoformat()
    date_end = dates.max().date().isoformat()
    official_sections, official_questions, sources = fetch_official(
        OireachtasClient(timeout_seconds=45, retries=5, backoff_seconds=2.0, sleep_seconds=0.1),
        date_start,
        date_end,
    )
    payload = {
        "date_start": date_start,
        "date_end": date_end,
        "sources": sources,
        "debate_sections": compare_keys(live_sections, official_sections, "debate_section_id", "snapshot_date"),
        "questions": compare_keys(live_questions, official_questions, "question_id", "question_date"),
    }
    write_report(payload, args.output_dir)
    print(json.dumps({
        "date_start": date_start,
        "date_end": date_end,
        "section_missing": payload["debate_sections"]["missing_from_live"],
        "section_extra": payload["debate_sections"]["extra_in_live"],
        "question_missing": payload["questions"]["missing_from_live"],
        "question_extra": payload["questions"]["extra_in_live"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
