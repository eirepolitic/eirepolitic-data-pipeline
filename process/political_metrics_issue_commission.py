#!/usr/bin/env python3
"""Commission first public issue measures from the promoted batch without publishing."""

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
from political_metrics.calculators.issues import (
    attach_issue_labels,
    grouped_issue_metrics,
    national_issue_metrics,
    party_issue_comparisons,
)
from political_metrics.commission import filter_period, prepare_eligible_td_speeches
from political_metrics.issue_audit import audit_issue_classification
from political_metrics.periods import resolve_period
from political_metrics.sources import canonical_speeches

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
OUT_DIR = Path(os.getenv("POLITICAL_METRICS_ISSUE_COMMISSION_DIR", "artifacts/political-metrics-issue-commission"))
PERIOD_SPEC = os.getenv("POLITICAL_METRICS_PERIOD", "2026-07")

TABLE_KEYS = {
    "silver_speeches": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
    "labels": "processed/oireachtas_unified/latest/csv/enrichment_speech_issue_labels.csv",
    "memberships": "processed/oireachtas_unified/latest/csv/silver_member_memberships.csv",
    "parties": "processed/oireachtas_unified/latest/csv/silver_member_parties.csv",
    "constituencies": "processed/oireachtas_unified/latest/csv/silver_member_constituencies.csv",
    "members": "processed/oireachtas_unified/latest/csv/silver_members.csv",
}


def _read_csv(s3, logical_key: str) -> tuple[pd.DataFrame, str]:
    resolved = resolve_production_key(s3, bucket=BUCKET, production_key=logical_key)
    obj = s3.get_object(Bucket=BUCKET, Key=resolved)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False), resolved


def _names(members: pd.DataFrame) -> pd.DataFrame:
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


def _main_issue(grouped: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if grouped.empty:
        return grouped
    ranked = grouped.sort_values([group_col, "issue_share", "issue_speech_count", "issue_label"], ascending=[True, False, False, True])
    return ranked.groupby(group_col, as_index=False).first()


def _markdown(report: dict) -> str:
    lines = [
        "# Political issue metrics commissioning review",
        "",
        f"**Period: {report['period']['start']} to {report['period']['end']}**",
        "",
        f"Policy-labelled Dáil speeches: **{report['classification']['policy_labelled_rows']:,}** of **{report['classification']['scope_rows']:,}** recorded speeches.",
        "",
        "## Main issues in recorded Dáil speeches",
        "",
    ]
    for idx, row in enumerate(report["top_national_issues"], 1):
        lines.append(f"{idx}. **{row['issue_label']}** — {row['issue_speech_count']:,} speeches ({row['issue_share'] * 100:.1f}% of policy-labelled speeches)")

    lines.extend(["", "## Strongest party emphasis compared with TDs overall", ""])
    for row in report["party_strongest_emphasis"]:
        lines.append(
            f"- **{row['party_name']}**: {row['issue_label']} — {row['issue_share'] * 100:.1f}% of its policy speeches, "
            f"**{row['share_vs_td_national_pp']:+.1f} percentage points** versus TDs overall"
        )

    lines.extend(["", "## Main issue for high-activity TDs", ""])
    for row in report["member_main_issues"]:
        lines.append(
            f"- **{row['member_name']}**: {row['issue_label']} — {row['issue_speech_count']} speeches "
            f"({row['issue_share'] * 100:.1f}% of their policy-labelled speeches)"
        )

    lines.extend(["", "## Main issue for high-activity constituencies", ""])
    for row in report["constituency_main_issues"]:
        lines.append(
            f"- **{row['constituency_name']}**: {row['issue_label']} — {row['issue_speech_count']} speeches "
            f"({row['issue_share'] * 100:.1f}% of policy-labelled speeches from its TDs)"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "These measures describe what recorded speeches were mainly about. They do not indicate whether a TD or party supported or opposed the issue, how important the contribution was, or whether speaking more often was more effective.",
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
    speeches = canonical_speeches(frames["silver_speeches"])
    period_speeches = filter_period(speeches, "debate_date", period)

    gate = audit_issue_classification(
        frames["silver_speeches"],
        frames["labels"],
        period_start=period.start.isoformat(),
        period_end=period.end.isoformat(),
    )
    if not gate.get("ready"):
        raise RuntimeError(f"issue classification gate failed: {gate}")

    period_all = attach_issue_labels(period_speeches, frames["labels"])
    eligible_td = prepare_eligible_td_speeches(
        speeches,
        frames["memberships"],
        frames["parties"],
        frames["constituencies"],
        period,
    )
    eligible_td = attach_issue_labels(eligible_td, frames["labels"])

    broad_national = national_issue_metrics(period_all)
    td_national = national_issue_metrics(eligible_td)
    member_issue = grouped_issue_metrics(eligible_td, group_col="member_code")
    party_issue = grouped_issue_metrics(eligible_td, group_col="party_uri")
    constituency_issue = grouped_issue_metrics(eligible_td, group_col="constituency_uri")

    party_names = _labels(frames["parties"], "party_uri", "party_name")
    independent_ids = set(party_names.loc[party_names["party_name"].eq("Independent"), "party_uri"].astype(str))
    party_comparison = party_issue_comparisons(
        party_issue,
        td_national,
        excluded_average_party_ids=independent_ids,
        baseline_min_policy_speeches=20,
    ).merge(party_names, on="party_uri", how="left")

    top_national = broad_national.sort_values(["issue_share", "issue_label"], ascending=[False, True]).head(10)

    reliable_party = party_comparison[party_comparison["comparison_public_safe"]].copy()
    party_best = (
        reliable_party.sort_values(["party_uri", "share_vs_td_national_pp", "issue_speech_count"], ascending=[True, False, False])
        .groupby("party_uri", as_index=False)
        .first()
        .sort_values("share_vs_td_national_pp", ascending=False)
    )

    member_names = _names(frames["members"])
    member_main = _main_issue(member_issue, "member_code").merge(member_names, on="member_code", how="left")
    member_totals = member_issue.groupby("member_code", as_index=False)["policy_speech_count"].max()
    member_main = member_main.drop(columns=["policy_speech_count"], errors="ignore").merge(member_totals, on="member_code", how="left")
    member_main = member_main.sort_values(["policy_speech_count", "issue_speech_count"], ascending=False).head(10)

    constituency_names = _labels(frames["constituencies"], "constituency_uri", "constituency_name")
    constituency_main = _main_issue(constituency_issue, "constituency_uri").merge(constituency_names, on="constituency_uri", how="left")
    constituency_totals = constituency_issue.groupby("constituency_uri", as_index=False)["policy_speech_count"].max()
    constituency_main = constituency_main.drop(columns=["policy_speech_count"], errors="ignore").merge(constituency_totals, on="constituency_uri", how="left")
    constituency_main = constituency_main.sort_values(["policy_speech_count", "issue_speech_count"], ascending=False).head(10)

    def records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
        output = frame[columns].copy()
        return output.where(pd.notna(output), None).to_dict(orient="records")

    report = {
        "commission_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_batch_id": str(pointer.get("batch_id") or pointer.get("mode") or "unknown"),
        "resolved_source_keys": resolved,
        "period": {"start": period.start.isoformat(), "end": period.end.isoformat()},
        "classification": gate,
        "td_policy_speech_count": int(eligible_td[eligible_td["issue_label"].ne("NONE")]["speech_id"].nunique()),
        "top_national_issues": records(top_national, ["issue_label", "issue_speech_count", "policy_speech_count", "issue_share"]),
        "party_strongest_emphasis": records(
            party_best,
            ["party_uri", "party_name", "issue_label", "issue_speech_count", "policy_speech_count", "issue_share", "td_national_issue_share", "share_vs_td_national_pp", "share_vs_average_party_pp", "reliability_status"],
        ),
        "member_main_issues": records(
            member_main,
            ["member_code", "member_name", "issue_label", "issue_speech_count", "policy_speech_count", "issue_share"],
        ),
        "constituency_main_issues": records(
            constituency_main,
            ["constituency_uri", "constituency_name", "issue_label", "issue_speech_count", "policy_speech_count", "issue_share"],
        ),
    }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(_markdown(report), encoding="utf-8")
    broad_national.to_csv(OUT_DIR / "national_issue_metrics.csv", index=False)
    party_comparison.to_csv(OUT_DIR / "party_issue_metrics.csv", index=False)
    member_issue.to_csv(OUT_DIR / "member_issue_metrics.csv", index=False)
    constituency_issue.to_csv(OUT_DIR / "constituency_issue_metrics.csv", index=False)
    print(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
