#!/usr/bin/env python3
"""Commission first public voting measures from the promoted batch without publishing."""

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
from political_metrics.calculators.votes import (
    constituency_vote_participation,
    eligible_division_pairs,
    member_vote_participation,
    party_vote_metrics,
)
from political_metrics.commission import filter_period
from political_metrics.periods import resolve_period

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
OUT_DIR = Path(os.getenv("POLITICAL_METRICS_VOTE_COMMISSION_DIR", "artifacts/political-metrics-vote-commission"))
PERIOD_SPEC = os.getenv("POLITICAL_METRICS_PERIOD", "2026-07")

TABLE_KEYS = {
    "divisions": "processed/oireachtas_unified/latest/csv/silver_divisions.csv",
    "member_votes": "processed/oireachtas_unified/latest/csv/silver_member_votes.csv",
    "memberships": "processed/oireachtas_unified/latest/csv/silver_member_memberships.csv",
    "parties": "processed/oireachtas_unified/latest/csv/silver_member_parties.csv",
    "constituencies": "processed/oireachtas_unified/latest/csv/silver_member_constituencies.csv",
    "members": "processed/oireachtas_unified/latest/csv/silver_members.csv",
}


def _read_csv(s3, logical_key: str) -> tuple[pd.DataFrame, str]:
    resolved = resolve_production_key(s3, bucket=BUCKET, production_key=logical_key)
    obj = s3.get_object(Bucket=BUCKET, Key=resolved)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""]), resolved


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


def _records(frame: pd.DataFrame, columns: list[str], *, n: int | None = None) -> list[dict]:
    data = frame[columns].copy()
    if n is not None:
        data = data.head(n)
    return data.where(pd.notna(data), None).to_dict(orient="records")


def _markdown(report: dict) -> str:
    lines = [
        "# Political voting metrics commissioning review",
        "",
        f"**Period: {report['period']['start']} to {report['period']['end']}**",
        "",
        f"Recorded Dáil divisions: **{report['division_count']}**",
        f"Recorded member votes: **{report['recorded_member_vote_count']:,}**",
        f"Recorded votes outside member eligibility: **{report['recorded_votes_outside_eligibility']}**",
        "",
        "## Highest recorded voting participation among TDs",
        "",
    ]
    for row in report["top_member_participation"]:
        lines.append(
            f"- **{row['member_name']}** — {row['votes_cast_count']} recorded votes from {row['eligible_division_count']} eligible divisions "
            f"({row['vote_participation_pct'] * 100:.1f}%)"
        )

    lines.extend(["", "## Party recorded voting participation", ""])
    for row in report["party_metrics"]:
        lines.append(
            f"- **{row['party_name']}** — {row['recorded_member_votes']:,} recorded votes from {row['eligible_member_divisions']:,} member-division opportunities "
            f"({row['vote_participation_pct'] * 100:.1f}%)"
        )

    lines.extend(["", "## Party voting unity", ""])
    for row in report["party_unity"]:
        lines.append(
            f"- **{row['party_name']}** — {row['vote_cohesion_pct'] * 100:.1f}% across {row['qualifying_unity_divisions']} qualifying divisions"
        )

    lines.extend(["", "## Constituency recorded voting participation", ""])
    for row in report["top_constituency_participation"]:
        lines.append(
            f"- **{row['constituency_name']}** — {row['vote_participation_pct'] * 100:.1f}% "
            f"({row['recorded_member_votes']}/{row['eligible_member_divisions']} member-division opportunities)"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "These figures describe recorded votes. A missing vote is not proof of physical absence. Party voting unity describes agreement among recorded party votes and is not a judgement about party discipline or the merits of dissent.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    pointer = read_json_required(s3, bucket=BUCKET, key=PRODUCTION_POINTER_KEY)

    frames: dict[str, pd.DataFrame] = {}
    resolved: dict[str, str] = {}
    for table, key in TABLE_KEYS.items():
        frames[table], resolved[table] = _read_csv(s3, key)

    period = resolve_period(PERIOD_SPEC)
    divisions = filter_period(frames["divisions"], "division_date", period)
    votes = filter_period(frames["member_votes"], "division_date", period)

    if divisions["division_id"].duplicated().any():
        raise RuntimeError("silver_divisions has duplicate division_id values in commissioning period")
    if votes.duplicated(subset=["division_id", "member_code"]).any():
        raise RuntimeError("silver_member_votes has duplicate member/division rows in commissioning period")
    valid_vote_codes = {"ta", "nil", "staon"}
    unexpected_codes = sorted(set(votes["vote_code"].dropna().astype(str)) - valid_vote_codes)
    if unexpected_codes:
        raise RuntimeError(f"unexpected vote codes in commissioning period: {unexpected_codes}")

    eligible = eligible_division_pairs(frames["memberships"], divisions)
    eligible_keys = set(zip(eligible["member_code"].astype(str), eligible["division_id"].astype(str)))
    vote_keys = list(zip(votes["member_code"].astype(str), votes["division_id"].astype(str)))
    outside = [key for key in vote_keys if key not in eligible_keys]
    if outside:
        raise RuntimeError(f"recorded votes outside member eligibility: {outside[:10]} (total={len(outside)})")

    member = member_vote_participation(votes, eligible).merge(_member_names(frames["members"]), on="member_code", how="left")
    party = party_vote_metrics(votes, eligible, frames["parties"]).merge(
        _labels(frames["parties"], "party_uri", "party_name"), on="party_uri", how="left"
    )
    constituency = constituency_vote_participation(votes, eligible, frames["constituencies"]).merge(
        _labels(frames["constituencies"], "constituency_uri", "constituency_name"), on="constituency_uri", how="left"
    )

    top_member = member.sort_values(
        ["vote_participation_pct", "eligible_division_count", "votes_cast_count", "member_name"],
        ascending=[False, False, False, True],
    ).head(10)
    party_sorted = party.sort_values(["vote_participation_pct", "party_name"], ascending=[False, True])
    party_unity = party[
        party["vote_cohesion_pct"].notna() & (party["qualifying_unity_divisions"] >= 5)
    ].sort_values(["vote_cohesion_pct", "qualifying_unity_divisions", "party_name"], ascending=[False, False, True])
    constituency_sorted = constituency.sort_values(
        ["vote_participation_pct", "eligible_member_divisions", "constituency_name"],
        ascending=[False, False, True],
    ).head(10)

    report = {
        "commission_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_batch_id": str(pointer.get("batch_id") or pointer.get("mode") or "unknown"),
        "resolved_source_keys": resolved,
        "period": {"start": period.start.isoformat(), "end": period.end.isoformat()},
        "division_count": int(divisions["division_id"].nunique()),
        "recorded_member_vote_count": int(len(votes)),
        "eligible_member_division_count": int(len(eligible)),
        "recorded_votes_outside_eligibility": len(outside),
        "vote_code_counts": {str(k): int(v) for k, v in votes["vote_code"].value_counts().to_dict().items()},
        "top_member_participation": _records(
            top_member,
            ["member_code", "member_name", "votes_cast_count", "eligible_division_count", "vote_participation_pct"],
        ),
        "party_metrics": _records(
            party_sorted,
            ["party_uri", "party_name", "recorded_member_votes", "eligible_member_divisions", "vote_participation_pct", "qualifying_unity_divisions", "vote_cohesion_pct"],
        ),
        "party_unity": _records(
            party_unity,
            ["party_uri", "party_name", "qualifying_unity_divisions", "unity_votes_aligned", "unity_votes_total", "vote_cohesion_pct"],
        ),
        "top_constituency_participation": _records(
            constituency_sorted,
            ["constituency_uri", "constituency_name", "recorded_member_votes", "eligible_member_divisions", "vote_participation_pct"],
        ),
    }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(_markdown(report), encoding="utf-8")
    member.to_csv(OUT_DIR / "member_vote_metrics.csv", index=False)
    party.to_csv(OUT_DIR / "party_vote_metrics.csv", index=False)
    constituency.to_csv(OUT_DIR / "constituency_vote_metrics.csv", index=False)
    print(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
