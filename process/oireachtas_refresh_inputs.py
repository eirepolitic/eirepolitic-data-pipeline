from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.schemas import load_table_registry


DEFAULT_TABLES = {
    "weekly": "silver_members,silver_member_memberships,silver_member_parties,silver_member_constituencies,silver_member_offices,silver_debate_records,silver_debate_sections,silver_speeches,silver_divisions,silver_division_tallies,silver_member_votes,silver_questions,gold_current_members,gold_member_activity_yearly,gold_member_activity_monthly,gold_content_fact_pool,control_pipeline_runs,control_table_manifests,control_data_quality_results",
    "monthly": "silver_constituencies,silver_parties,silver_source_files,silver_bills,silver_bill_versions,silver_bill_stages,silver_bill_related_docs,silver_bill_sponsors,silver_bill_debates,silver_bill_events,gold_constituency_activity_yearly,gold_content_fact_pool,control_pipeline_runs,control_table_manifests,control_data_quality_results",
    "yearly": "silver_houses,silver_constituencies,silver_parties,silver_members,silver_member_memberships,silver_member_parties,silver_member_constituencies,silver_member_offices,silver_bills,silver_bill_versions,silver_bill_stages,gold_current_members,gold_member_activity_yearly,gold_constituency_activity_yearly,gold_content_fact_pool,control_pipeline_runs,control_table_manifests,control_data_quality_results",
}
DEFAULT_MODES = {"weekly": "incremental", "monthly": "incremental", "yearly": "full"}
DEFAULT_PAGE_SIZES = {"weekly": 100, "monthly": 200, "yearly": 200}
VALID_MODES = {
    "weekly": {"test", "incremental", "full"},
    "monthly": {"test", "incremental", "full"},
    "yearly": {"test", "full", "backfill"},
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize and validate Oireachtas refresh inputs.")
    p.add_argument("--refresh-type", required=True, choices=sorted(DEFAULT_TABLES))
    p.add_argument("--mode", default="")
    p.add_argument("--tables", default="")
    p.add_argument("--chamber", default="dail")
    p.add_argument("--house-no", default="34")
    p.add_argument("--date-start", default="")
    p.add_argument("--date-end", default="")
    p.add_argument("--page-size", default="")
    p.add_argument("--sample-rows", default="10")
    p.add_argument("--as-of-date", default="")
    p.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT", ""))
    return p


def normalize(args: argparse.Namespace) -> dict[str, object]:
    refresh_type = args.refresh_type
    as_of = _parse_date(args.as_of_date) if args.as_of_date else date.today()
    mode = (args.mode or DEFAULT_MODES[refresh_type]).strip()
    if mode not in VALID_MODES[refresh_type]:
        raise ValueError(f"mode {mode!r} is invalid for {refresh_type}; expected one of {sorted(VALID_MODES[refresh_type])}")

    chamber = args.chamber.strip().lower()
    if chamber not in {"dail", "seanad"}:
        raise ValueError("chamber must be dail or seanad")
    house_no = args.house_no.strip()
    if not house_no.isdigit() or int(house_no) <= 0:
        raise ValueError("house_no must be a positive integer")

    default_start, default_end = _default_window(refresh_type, as_of)
    date_start = _parse_date(args.date_start).isoformat() if args.date_start else default_start.isoformat()
    date_end = _parse_date(args.date_end).isoformat() if args.date_end else default_end.isoformat()
    if date_start > date_end:
        raise ValueError(f"date_start {date_start} must not be after date_end {date_end}")

    page_size = _bounded_int(args.page_size or DEFAULT_PAGE_SIZES[refresh_type], name="page_size", minimum=1, maximum=200)
    sample_rows = _bounded_int(args.sample_rows, name="sample_rows", minimum=1, maximum=100)

    registry = load_table_registry()
    requested = [item.strip() for item in (args.tables or DEFAULT_TABLES[refresh_type]).split(",") if item.strip()]
    if not requested:
        raise ValueError("at least one table is required")
    duplicates = sorted({table for table in requested if requested.count(table) > 1})
    unknown = sorted(set(requested) - set(registry))
    if duplicates:
        raise ValueError(f"duplicate tables are not allowed: {duplicates}")
    if unknown:
        raise ValueError(f"unknown tables: {unknown}")

    return {
        "refresh_type": refresh_type,
        "mode": mode,
        "tables": ",".join(requested),
        "chamber": chamber,
        "house_no": house_no,
        "date_start": date_start,
        "date_end": date_end,
        "page_size": page_size,
        "sample_rows": sample_rows,
        "table_count": len(requested),
    }


def _default_window(refresh_type: str, as_of: date) -> tuple[date, date]:
    if refresh_type == "weekly":
        return as_of - timedelta(days=35), as_of
    if refresh_type == "monthly":
        month_start = as_of.replace(day=1)
        previous_end = month_start - timedelta(days=1)
        previous_start = previous_end.replace(day=1)
        return previous_start - timedelta(days=7), previous_end
    previous_year = as_of.year - 1
    return date(previous_year, 1, 1), date(previous_year, 12, 31)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def _bounded_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def write_outputs(payload: dict[str, object], path: str) -> None:
    if not path:
        return
    output = Path(path)
    with output.open("a", encoding="utf-8") as handle:
        for key, value in payload.items():
            handle.write(f"{key}={value}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = normalize(args)
    write_outputs(payload, args.github_output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
