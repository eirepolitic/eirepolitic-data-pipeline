#!/usr/bin/env python3
"""Commission first public parliamentary-question measures without publishing."""

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
from political_metrics.calculators.questions import (
    grouped_question_metrics,
    member_question_metrics,
    national_question_metrics,
    prepare_eligible_td_questions,
    question_type_distribution,
    recipient_distribution,
)
from political_metrics.commission import filter_period
from political_metrics.periods import resolve_period

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
OUT_DIR = Path(os.getenv("POLITICAL_METRICS_QUESTION_COMMISSION_DIR", "artifacts/political-metrics-question-commission"))
PERIOD_SPEC = os.getenv("POLITICAL_METRICS_PERIOD", "2026-07")

TABLE_KEYS = {
    "questions": "processed/oireachtas_unified/latest/csv/silver_questions.csv",
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


def _records(frame: pd.DataFrame, columns: list[str], n: int | None = None) -> list[dict]:
    data = frame[columns].copy()
    if n is not None:
        data = data.head(n)
    return data.where(pd.notna(data), None).to_dict(orient="records")


def _markdown(report: dict) -> str:
    n = report["national"]
    lines = [
        "# Parliamentary question metrics commissioning review",
        "",
        f"**Period: {report['period']['start']} to {report['period']['end']}**",
        "",
        f"Recorded parliamentary questions: **{n['question_count']:,}**",
        f"TDs submitting questions: **{n['asking_member_count']}**",
        f"Question dates: **{n['question_day_count']}**",
        f"Recorded question types: **{n['question_type_count']}**",
        f"Recorded ministers/departments questioned: **{n['recipient_count']}**",
        "",
        "## Highest question counts by TD",
        "",
    ]
    for row in report["top_members"]:
        lines.append(
            f"- **{row['member_name']}** — {row['question_count']} questions across {row['question_day_count']} dates; "
            f"{row['recipient_count']} recorded ministers/departments questioned"
        )

    lines.extend(["", "## Party question totals", ""])
    for row in report["party_metrics"]:
        lines.append(
            f"- **{row['party_name']}** — {row['question_count']} questions from {row['asking_member_count']} TDs; "
            f"{row['recipient_count']} recorded ministers/departments questioned"
        )

    lines.extend(["", "## Highest constituency question totals", ""])
    for row in report["top_constituencies"]:
        lines.append(
            f"- **{row['constituency_name']}** — {row['question_count']} questions from {row['asking_member_count']} TDs"
        )

    lines.extend(["", "## Question types", ""])
    for row in report["question_types"]:
        lines.append(f"- **{row['question_type']}** — {row['question_count']} ({row['question_type_share'] * 100:.1f}%)")

    lines.extend(["", "## Most-questioned recorded recipients", ""])
    for row in report["top_recipients"]:
        lines.append(
            f"- **{row['to_minister_or_department']}** — {row['question_count']} questions ({row['question_share'] * 100:.1f}%)"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "These figures measure recorded parliamentary-question activity. They do not show whether a question was effective, important, answered satisfactorily, or raised elsewhere through another parliamentary mechanism.",
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
    questions = filter_period(frames["questions"], "question_date", period)
    if questions["question_id"].duplicated().any():
        raise RuntimeError("silver_questions contains duplicate question_id values in commissioning period")
    required_populated = ["question_id", "question_date", "asked_by_member_code", "question_text", "to_minister_or_department"]
    for col in required_populated:
        if questions[col].isna().any() or questions[col].fillna("").astype(str).str.strip().eq("").any():
            raise RuntimeError(f"silver_questions contains blank {col} values in commissioning period")

    eligible = prepare_eligible_td_questions(
        questions,
        frames["memberships"],
        frames["parties"],
        frames["constituencies"],
    )
    raw_ids = set(questions["question_id"].astype(str))
    eligible_ids = set(eligible["question_id"].astype(str))
    excluded_ids = sorted(raw_ids - eligible_ids)
    if excluded_ids:
        raise RuntimeError(f"questions could not be attributed to active TD membership: {excluded_ids[:10]} (total={len(excluded_ids)})")
    if eligible["party_uri"].isna().any():
        raise RuntimeError("one or more eligible TD questions lack historical party attribution")
    if eligible["constituency_uri"].isna().any():
        raise RuntimeError("one or more eligible TD questions lack historical constituency attribution")

    national = national_question_metrics(eligible)
    member = member_question_metrics(eligible).merge(_member_names(frames["members"]), on="member_code", how="left")
    party = grouped_question_metrics(eligible, group_col="party_uri").merge(
        _labels(frames["parties"], "party_uri", "party_name"), on="party_uri", how="left"
    )
    constituency = grouped_question_metrics(eligible, group_col="constituency_uri").merge(
        _labels(frames["constituencies"], "constituency_uri", "constituency_name"), on="constituency_uri", how="left"
    )
    qtypes = question_type_distribution(eligible).sort_values(["question_count", "question_type"], ascending=[False, True])
    recipients = recipient_distribution(eligible).sort_values(["question_count", "to_minister_or_department"], ascending=[False, True])

    member_sorted = member.sort_values(["question_count", "question_day_count", "member_name"], ascending=[False, False, True]).head(10)
    party_sorted = party.sort_values(["question_count", "party_name"], ascending=[False, True])
    constituency_sorted = constituency.sort_values(["question_count", "constituency_name"], ascending=[False, True]).head(10)

    report = {
        "commission_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_batch_id": str(pointer.get("batch_id") or pointer.get("mode") or "unknown"),
        "resolved_source_keys": resolved,
        "period": {"start": period.start.isoformat(), "end": period.end.isoformat()},
        "source_question_count": int(questions["question_id"].nunique()),
        "eligible_question_count": int(eligible["question_id"].nunique()),
        "excluded_question_count": len(excluded_ids),
        "national": national,
        "top_members": _records(member_sorted, ["member_code", "member_name", "question_count", "question_day_count", "question_type_count", "recipient_count"]),
        "party_metrics": _records(party_sorted, ["party_uri", "party_name", "question_count", "asking_member_count", "question_day_count", "question_type_count", "recipient_count"]),
        "top_constituencies": _records(constituency_sorted, ["constituency_uri", "constituency_name", "question_count", "asking_member_count", "question_day_count", "question_type_count", "recipient_count"]),
        "question_types": _records(qtypes, ["question_type", "question_count", "question_type_share"]),
        "top_recipients": _records(recipients, ["to_minister_or_department", "question_count", "question_share"], n=10),
    }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(_markdown(report), encoding="utf-8")
    member.to_csv(OUT_DIR / "member_question_metrics.csv", index=False)
    party.to_csv(OUT_DIR / "party_question_metrics.csv", index=False)
    constituency.to_csv(OUT_DIR / "constituency_question_metrics.csv", index=False)
    qtypes.to_csv(OUT_DIR / "question_type_metrics.csv", index=False)
    recipients.to_csv(OUT_DIR / "question_recipient_metrics.csv", index=False)
    print(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
