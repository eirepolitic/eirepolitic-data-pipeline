#!/usr/bin/env python3
"""Commission the first political speech metrics from the promoted batch without publishing."""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3
import pandas as pd

from extract.oireachtas.batch import PRODUCTION_POINTER_KEY, read_json_required, resolve_production_key
from political_metrics.commission import calculate_core_speech_metrics, reconciliation_checks
from political_metrics.periods import resolve_period
from political_metrics.sources import canonical_speeches

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
OUT_DIR = Path(os.getenv("POLITICAL_METRICS_COMMISSION_DIR", "artifacts/political-metrics-commission"))
PERIOD_SPEC = os.getenv("POLITICAL_METRICS_PERIOD", "last_completed_month")

TABLE_KEYS = {
    "silver_speeches": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
    "silver_member_memberships": "processed/oireachtas_unified/latest/csv/silver_member_memberships.csv",
    "silver_member_parties": "processed/oireachtas_unified/latest/csv/silver_member_parties.csv",
    "silver_member_constituencies": "processed/oireachtas_unified/latest/csv/silver_member_constituencies.csv",
    "silver_debate_records": "processed/oireachtas_unified/latest/csv/silver_debate_records.csv",
    "silver_members": "processed/oireachtas_unified/latest/csv/silver_members.csv",
}


def _read_csv(s3, logical_key: str) -> tuple[pd.DataFrame, str]:
    resolved_key = resolve_production_key(s3, bucket=BUCKET, production_key=logical_key)
    obj = s3.get_object(Bucket=BUCKET, Key=resolved_key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""]), resolved_key


def _member_names(members: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["member_code", "full_name", "display_name", "first_name", "last_name"] if col in members.columns]
    data = members[cols].drop_duplicates("member_code").copy()
    if "full_name" in data.columns:
        data["member_name"] = data["full_name"]
    elif "display_name" in data.columns:
        data["member_name"] = data["display_name"]
    else:
        first = data["first_name"].fillna("") if "first_name" in data.columns else ""
        last = data["last_name"].fillna("") if "last_name" in data.columns else ""
        data["member_name"] = (first + " " + last).str.strip()
    return data[["member_code", "member_name"]]


def _labels(history: pd.DataFrame, id_col: str, name_col: str) -> pd.DataFrame:
    return history[[id_col, name_col]].dropna(subset=[id_col]).drop_duplicates(id_col, keep="last")


def _top_rows(frame: pd.DataFrame, *, sort_col: str, columns: list[str], n: int = 10) -> list[dict]:
    if frame.empty:
        return []
    data = frame.sort_values(sort_col, ascending=False).head(n)[columns].copy()
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].round(3)
    return data.where(pd.notna(data), None).to_dict(orient="records")


def _markdown(report: dict) -> str:
    r = report["reconciliation"]
    lines = [
        "# Political metrics commissioning review",
        "",
        f"**Period: {report['period']['start']} to {report['period']['end']}**",
        "",
        "This is a non-publishing commissioning run. The figures below were calculated from the promoted canonical data but were not written back as production metrics.",
        "",
        "## Overall activity",
        "",
        f"- Recorded Dáil speeches: **{r['national_distinct_speeches']:,}**",
        f"- Speeches by eligible TDs: **{r['eligible_td_distinct_speeches']:,}**",
        f"- Debate days: **{report['national']['debate_day_count']}**",
        f"- Speeches per debate day: **{report['national']['speeches_per_debate_day']:.1f}**" if report['national']['speeches_per_debate_day'] is not None else "- Speeches per debate day: unavailable",
        "",
        "## Most speeches by TD",
        "",
    ]
    for idx, row in enumerate(report["top_members"], 1):
        lines.append(f"{idx}. **{row['member_name']}** — {row['speech_count']:,} speeches across {row['speaking_day_count']} speaking days")

    lines.extend(["", "## Party speaking activity", ""])
    for idx, row in enumerate(report["top_parties"], 1):
        rate = row.get("speeches_per_active_member")
        rate_text = f"; {rate:.1f} speeches per period-adjusted TD" if rate is not None else ""
        lines.append(f"{idx}. **{row['party_name']}** — {row['speech_count']:,} speeches{rate_text}")

    lines.extend(["", "## Constituency speaking activity", ""])
    for idx, row in enumerate(report["top_constituencies"], 1):
        rate = row.get("speeches_per_active_member")
        rate_text = f"; {rate:.1f} speeches per period-adjusted representative" if rate is not None else ""
        lines.append(f"{idx}. **{row['constituency_name']}** — {row['speech_count']:,} speeches{rate_text}")

    lines.extend([
        "",
        "## Reconciliation",
        "",
        f"- Member totals match eligible TD speeches: **{'PASS' if r['member_sum_matches_eligible_td_speeches'] else 'FAIL'}**",
        f"- Party totals match eligible TD speeches: **{'PASS' if r['party_sum_matches_eligible_td_speeches'] else 'FAIL'}**",
        f"- Constituency totals match eligible TD speeches: **{'PASS' if r['constituency_sum_matches_eligible_td_speeches'] else 'FAIL'}**",
        f"- National calculator matches source total: **{'PASS' if r['national_calculator_matches'] else 'FAIL'}**",
        "",
        "These are activity measures. They do not measure political effectiveness, attendance, influence, or quality of contribution.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    pointer = read_json_required(s3, bucket=BUCKET, key=PRODUCTION_POINTER_KEY)

    frames: dict[str, pd.DataFrame] = {}
    keys: dict[str, str] = {}
    for table, logical_key in TABLE_KEYS.items():
        frames[table], keys[table] = _read_csv(s3, logical_key)

    period = resolve_period(PERIOD_SPEC, today=datetime.now(timezone.utc))
    speeches = canonical_speeches(frames["silver_speeches"])
    results = calculate_core_speech_metrics(
        speeches=speeches,
        memberships=frames["silver_member_memberships"],
        member_parties=frames["silver_member_parties"],
        member_constituencies=frames["silver_member_constituencies"],
        debate_records=frames["silver_debate_records"],
        period=period,
    )
    reconciliation = reconciliation_checks(results)

    member = results["member_metrics"].merge(_member_names(frames["silver_members"]), on="member_code", how="left")
    party = results["party_metrics"].merge(
        _labels(frames["silver_member_parties"], "party_uri", "party_name"), on="party_uri", how="left"
    )
    constituency = results["constituency_metrics"].merge(
        _labels(frames["silver_member_constituencies"], "constituency_uri", "constituency_name"), on="constituency_uri", how="left"
    )

    national = results["national_metrics"]
    report = {
        "commission_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_batch_id": str(pointer.get("batch_id") or pointer.get("mode") or "unknown"),
        "resolved_source_keys": keys,
        "period": {"spec": PERIOD_SPEC, "start": period.start.isoformat(), "end": period.end.isoformat()},
        "national": national,
        "reconciliation": reconciliation,
        "top_members": _top_rows(
            member,
            sort_col="speech_count",
            columns=["member_code", "member_name", "speech_count", "speaking_day_count", "speeches_per_eligible_debate_day"],
        ),
        "top_parties": _top_rows(
            party,
            sort_col="speech_count",
            columns=["party_uri", "party_name", "speech_count", "speaking_member_count", "active_member_equivalent", "speeches_per_active_member"],
        ),
        "top_constituencies": _top_rows(
            constituency,
            sort_col="speech_count",
            columns=["constituency_uri", "constituency_name", "speech_count", "speaking_member_count", "active_member_equivalent", "speeches_per_active_member"],
        ),
    }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(_markdown(report), encoding="utf-8")
    member.to_csv(OUT_DIR / "member_metrics.csv", index=False)
    party.to_csv(OUT_DIR / "party_metrics.csv", index=False)
    constituency.to_csv(OUT_DIR / "constituency_metrics.csv", index=False)

    print(_markdown(report))
    if not all(
        reconciliation[key]
        for key in [
            "national_calculator_matches",
            "member_sum_matches_eligible_td_speeches",
            "party_sum_matches_eligible_td_speeches",
            "constituency_sum_matches_eligible_td_speeches",
        ]
    ):
        raise SystemExit("commissioning reconciliation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
