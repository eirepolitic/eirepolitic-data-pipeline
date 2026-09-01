#!/usr/bin/env python3
"""Commission Option A metric materialization locally without S3 publication."""

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
    reliability_status,
)
from political_metrics.calculators.questions import (
    grouped_question_metrics,
    member_question_metrics,
    prepare_eligible_td_questions,
    question_type_distribution,
    recipient_distribution,
)
from political_metrics.calculators.speeches import grouped_speech_metrics, member_speech_metrics, national_speech_metrics
from political_metrics.calculators.votes import (
    constituency_vote_participation,
    eligible_division_pairs,
    member_vote_participation,
    party_vote_metrics,
)
from political_metrics.commission import filter_period, prepare_eligible_td_speeches
from political_metrics.eligibility import constituency_debate_day_exposure, member_debate_day_exposure, party_debate_day_exposure
from political_metrics.foundations import (
    build_daily_activity_components,
    build_daily_issue_activity,
    build_daily_question_dimensions,
    build_division_party_vote_components,
)
from political_metrics.issue_audit import audit_issue_classification
from political_metrics.materialize import get_dataset_contract, load_materialization_contract, write_materialized_dataset
from political_metrics.periods import resolve_period
from political_metrics.results import append_metric_rows, metric_result_row, metric_results_frame
from political_metrics.sources import canonical_speeches

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
OUT_DIR = Path(os.getenv("POLITICAL_METRICS_MATERIALIZATION_DIR", "artifacts/political-metrics-materialization"))
PERIOD_SPEC = os.getenv("POLITICAL_METRICS_PERIOD", "2026-07")
CONTRACT_PATH = REPO_ROOT / "configs/political_metrics/materialization.yml"

TABLE_KEYS = {
    "speeches": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
    "labels": "processed/oireachtas_unified/latest/csv/enrichment_speech_issue_labels.csv",
    "memberships": "processed/oireachtas_unified/latest/csv/silver_member_memberships.csv",
    "parties": "processed/oireachtas_unified/latest/csv/silver_member_parties.csv",
    "constituencies": "processed/oireachtas_unified/latest/csv/silver_member_constituencies.csv",
    "debates": "processed/oireachtas_unified/latest/csv/silver_debate_records.csv",
    "divisions": "processed/oireachtas_unified/latest/csv/silver_divisions.csv",
    "votes": "processed/oireachtas_unified/latest/csv/silver_member_votes.csv",
    "questions": "processed/oireachtas_unified/latest/csv/silver_questions.csv",
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


def _reliability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["reliability_status"] = result["policy_speech_count"].map(lambda value: reliability_status(int(value)))
    result["public_use_status"] = result["reliability_status"].map(
        lambda value: "not_certified" if value == "insufficient_for_comparison" else "suitable_with_context"
    )
    result["warning_code"] = result["reliability_status"].map(
        {"reliable": "none", "caution": "small_sample_caution", "insufficient_for_comparison": "small_sample_suppressed"}
    )
    return result


def _monthly_results(
    *,
    frames: dict[str, pd.DataFrame],
    period,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    speeches = canonical_speeches(frames["speeches"])
    period_speeches = filter_period(speeches, "debate_date", period)
    period_debates = filter_period(frames["debates"], "debate_date", period)
    debate_days = pd.to_datetime(period_debates["debate_date"], errors="coerce").dropna().dt.normalize().drop_duplicates().tolist()
    eligible_speeches = prepare_eligible_td_speeches(
        speeches, frames["memberships"], frames["parties"], frames["constituencies"], period
    )

    member_names = _member_names(frames["members"])
    party_names = _labels(frames["parties"], "party_uri", "party_name")
    constituency_names = _labels(frames["constituencies"], "constituency_uri", "constituency_name")

    member_exposure = member_debate_day_exposure(frames["memberships"], debate_days)
    member_exposure = member_exposure[member_exposure["eligible_debate_days"] > 0]
    party_exposure = party_debate_day_exposure(frames["parties"], debate_days)
    constituency_exposure = constituency_debate_day_exposure(frames["constituencies"], debate_days)

    member_speech = member_speech_metrics(eligible_speeches, member_exposure).merge(member_names, on="member_code", how="left")
    party_speech = grouped_speech_metrics(eligible_speeches, group_col="party_uri", group_exposure=party_exposure).merge(party_names, on="party_uri", how="left")
    constituency_speech = grouped_speech_metrics(eligible_speeches, group_col="constituency_uri", group_exposure=constituency_exposure).merge(constituency_names, on="constituency_uri", how="left")
    national_speech = national_speech_metrics(period_speeches, debate_days=debate_days)

    gate = audit_issue_classification(
        frames["speeches"], frames["labels"], period_start=period.start.isoformat(), period_end=period.end.isoformat()
    )
    if not gate.get("ready"):
        raise RuntimeError(f"issue classification gate failed for materialization: {gate}")
    period_all_issues = attach_issue_labels(period_speeches, frames["labels"])
    td_issues = attach_issue_labels(eligible_speeches, frames["labels"])
    national_issue = national_issue_metrics(period_all_issues)
    td_national_issue = national_issue_metrics(td_issues)
    member_issue = _reliability_columns(grouped_issue_metrics(td_issues, group_col="member_code").merge(member_names, on="member_code", how="left"))
    party_issue = _reliability_columns(grouped_issue_metrics(td_issues, group_col="party_uri").merge(party_names, on="party_uri", how="left"))
    constituency_issue = _reliability_columns(grouped_issue_metrics(td_issues, group_col="constituency_uri").merge(constituency_names, on="constituency_uri", how="left"))
    independent_ids = set(party_names.loc[party_names["party_name"].eq("Independent"), "party_uri"].astype(str))
    party_compare = party_issue_comparisons(
        grouped_issue_metrics(td_issues, group_col="party_uri"),
        td_national_issue,
        excluded_average_party_ids=independent_ids,
        baseline_min_policy_speeches=20,
    ).merge(party_names, on="party_uri", how="left")
    party_compare["public_use_status"] = party_compare["comparison_public_safe"].map(lambda safe: "suitable_with_context" if safe else "not_certified")
    party_compare["warning_code"] = party_compare["reliability_status"].map(
        {"reliable": "none", "caution": "small_sample_caution", "insufficient_for_comparison": "small_sample_suppressed"}
    )

    divisions = filter_period(frames["divisions"], "division_date", period)
    votes = filter_period(frames["votes"], "division_date", period)
    eligible_pairs = eligible_division_pairs(frames["memberships"], divisions)
    member_vote = member_vote_participation(votes, eligible_pairs).merge(member_names, on="member_code", how="left")
    party_vote = party_vote_metrics(votes, eligible_pairs, frames["parties"]).merge(party_names, on="party_uri", how="left")
    constituency_vote = constituency_vote_participation(votes, eligible_pairs, frames["constituencies"]).merge(constituency_names, on="constituency_uri", how="left")

    questions = filter_period(frames["questions"], "question_date", period)
    eligible_questions = prepare_eligible_td_questions(
        questions, frames["memberships"], frames["parties"], frames["constituencies"]
    )
    member_question = member_question_metrics(eligible_questions).merge(member_names, on="member_code", how="left")
    party_question = grouped_question_metrics(eligible_questions, group_col="party_uri").merge(party_names, on="party_uri", how="left")
    constituency_question = grouped_question_metrics(eligible_questions, group_col="constituency_uri").merge(constituency_names, on="constituency_uri", how="left")
    question_types = question_type_distribution(eligible_questions)
    recipients = recipient_distribution(eligible_questions)

    rows: list[dict] = []
    ps, pe = period.start.isoformat(), period.end.isoformat()

    append_metric_rows(rows, member_speech, metric_id="member_speech_count", metric_version=1, value_col="speech_count", numerator_col="speech_count", denominator_col=None, grain="member", entity_id_col="member_code", entity_name_col="member_name", period_start=ps, period_end=pe, output_unit="count", source_batch_id=source_batch_id, contract_version=contract_version)
    append_metric_rows(rows, member_speech, metric_id="member_speaking_day_count", metric_version=1, value_col="speaking_day_count", numerator_col="speaking_day_count", denominator_col=None, grain="member", entity_id_col="member_code", entity_name_col="member_name", period_start=ps, period_end=pe, output_unit="days", source_batch_id=source_batch_id, contract_version=contract_version)
    append_metric_rows(rows, member_speech, metric_id="member_speeches_per_eligible_debate_day", metric_version=1, value_col="speeches_per_eligible_debate_day", numerator_col="speech_count", denominator_col="eligible_debate_days", grain="member", entity_id_col="member_code", entity_name_col="member_name", period_start=ps, period_end=pe, output_unit="speeches_per_day", source_batch_id=source_batch_id, contract_version=contract_version)

    append_metric_rows(rows, party_speech, metric_id="party_speech_count", metric_version=1, value_col="speech_count", numerator_col="speech_count", denominator_col=None, grain="party", entity_id_col="party_uri", entity_name_col="party_name", period_start=ps, period_end=pe, output_unit="count", source_batch_id=source_batch_id, contract_version=contract_version)
    append_metric_rows(rows, party_speech, metric_id="party_speeches_per_active_member", metric_version=1, value_col="speeches_per_active_member", numerator_col="speech_count", denominator_col="active_member_equivalent", grain="party", entity_id_col="party_uri", entity_name_col="party_name", period_start=ps, period_end=pe, output_unit="speeches_per_active_member", source_batch_id=source_batch_id, contract_version=contract_version)
    append_metric_rows(rows, constituency_speech, metric_id="constituency_speech_count", metric_version=1, value_col="speech_count", numerator_col="speech_count", denominator_col=None, grain="constituency", entity_id_col="constituency_uri", entity_name_col="constituency_name", period_start=ps, period_end=pe, output_unit="count", source_batch_id=source_batch_id, contract_version=contract_version)
    append_metric_rows(rows, constituency_speech, metric_id="constituency_speeches_per_active_rep", metric_version=1, value_col="speeches_per_active_member", numerator_col="speech_count", denominator_col="active_member_equivalent", grain="constituency", entity_id_col="constituency_uri", entity_name_col="constituency_name", period_start=ps, period_end=pe, output_unit="speeches_per_active_representative", source_batch_id=source_batch_id, contract_version=contract_version)

    rows.append(metric_result_row(metric_id="national_speech_count", metric_version=1, period_start=ps, period_end=pe, grain="national", entity_id="dail", entity_name="Dáil", value=national_speech["speech_count"], numerator=national_speech["speech_count"], denominator=None, output_unit="count", source_batch_id=source_batch_id, contract_version=contract_version))
    rows.append(metric_result_row(metric_id="national_speeches_per_debate_day", metric_version=1, period_start=ps, period_end=pe, grain="national", entity_id="dail", entity_name="Dáil", value=national_speech["speeches_per_debate_day"], numerator=national_speech["speech_count"], denominator=national_speech["debate_day_count"], output_unit="speeches_per_day", source_batch_id=source_batch_id, contract_version=contract_version))

    national_issue["entity_id"] = "dail"
    national_issue["entity_name"] = "Dáil"
    append_metric_rows(rows, national_issue, metric_id="national_issue_speech_count", metric_version=1, value_col="issue_speech_count", numerator_col="issue_speech_count", denominator_col=None, grain="national", entity_id_col="entity_id", entity_name_col="entity_name", period_start=ps, period_end=pe, output_unit="count", source_batch_id=source_batch_id, contract_version=contract_version, dimension_name="issue", dimension_value_col="issue_label")
    append_metric_rows(rows, national_issue, metric_id="national_issue_share", metric_version=1, value_col="issue_share", numerator_col="issue_speech_count", denominator_col="policy_speech_count", grain="national", entity_id_col="entity_id", entity_name_col="entity_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version, dimension_name="issue", dimension_value_col="issue_label")

    append_metric_rows(rows, member_issue, metric_id="member_issue_share", metric_version=1, value_col="issue_share", numerator_col="issue_speech_count", denominator_col="policy_speech_count", grain="member", entity_id_col="member_code", entity_name_col="member_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version, dimension_name="issue", dimension_value_col="issue_label", reliability_col="reliability_status", public_use_col="public_use_status", warning_col="warning_code")
    append_metric_rows(rows, party_issue, metric_id="party_issue_share", metric_version=1, value_col="issue_share", numerator_col="issue_speech_count", denominator_col="policy_speech_count", grain="party", entity_id_col="party_uri", entity_name_col="party_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version, dimension_name="issue", dimension_value_col="issue_label", reliability_col="reliability_status", public_use_col="public_use_status", warning_col="warning_code")
    append_metric_rows(rows, constituency_issue, metric_id="constituency_issue_share", metric_version=1, value_col="issue_share", numerator_col="issue_speech_count", denominator_col="policy_speech_count", grain="constituency", entity_id_col="constituency_uri", entity_name_col="constituency_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version, dimension_name="issue", dimension_value_col="issue_label", reliability_col="reliability_status", public_use_col="public_use_status", warning_col="warning_code")
    append_metric_rows(rows, party_compare, metric_id="party_issue_share_vs_td_national_pp", metric_version=1, value_col="share_vs_td_national_pp", numerator_col="issue_share", denominator_col="td_national_issue_share", grain="party", entity_id_col="party_uri", entity_name_col="party_name", period_start=ps, period_end=pe, output_unit="percentage_points", source_batch_id=source_batch_id, contract_version=contract_version, dimension_name="issue", dimension_value_col="issue_label", reliability_col="reliability_status", public_use_col="public_use_status", warning_col="warning_code")
    append_metric_rows(rows, party_compare, metric_id="party_issue_share_vs_average_party_pp", metric_version=1, value_col="share_vs_average_party_pp", numerator_col="issue_share", denominator_col="average_party_issue_share", grain="party", entity_id_col="party_uri", entity_name_col="party_name", period_start=ps, period_end=pe, output_unit="percentage_points", source_batch_id=source_batch_id, contract_version=contract_version, dimension_name="issue", dimension_value_col="issue_label", reliability_col="reliability_status", public_use_col="public_use_status", warning_col="warning_code")

    append_metric_rows(rows, member_vote, metric_id="member_vote_participation_pct", metric_version=1, value_col="vote_participation_pct", numerator_col="votes_cast_count", denominator_col="eligible_division_count", grain="member", entity_id_col="member_code", entity_name_col="member_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version)
    append_metric_rows(rows, party_vote, metric_id="party_vote_participation_pct", metric_version=1, value_col="vote_participation_pct", numerator_col="recorded_member_votes", denominator_col="eligible_member_divisions", grain="party", entity_id_col="party_uri", entity_name_col="party_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version)
    append_metric_rows(rows, constituency_vote, metric_id="constituency_vote_participation_pct", metric_version=1, value_col="vote_participation_pct", numerator_col="recorded_member_votes", denominator_col="eligible_member_divisions", grain="constituency", entity_id_col="constituency_uri", entity_name_col="constituency_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version)
    party_vote["public_use_status"] = party_vote["unity_public_safe"].map(lambda safe: "suitable_with_context" if safe else "not_certified")
    party_vote["warning_code"] = party_vote["unity_reliability_status"].map({"reliable": "none", "caution": "small_sample_caution", "insufficient_for_comparison": "small_sample_suppressed"})
    append_metric_rows(rows, party_vote, metric_id="party_vote_cohesion_pct", metric_version=1, value_col="vote_cohesion_pct", numerator_col="unity_votes_aligned", denominator_col="unity_votes_total", grain="party", entity_id_col="party_uri", entity_name_col="party_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version, reliability_col="unity_reliability_status", public_use_col="public_use_status", warning_col="warning_code")
    rows.append(metric_result_row(metric_id="national_division_count", metric_version=1, period_start=ps, period_end=pe, grain="national", entity_id="dail", entity_name="Dáil", value=int(divisions["division_id"].nunique()), numerator=int(divisions["division_id"].nunique()), denominator=None, output_unit="count", source_batch_id=source_batch_id, contract_version=contract_version))

    append_metric_rows(rows, member_question, metric_id="member_question_count", metric_version=1, value_col="question_count", numerator_col="question_count", denominator_col=None, grain="member", entity_id_col="member_code", entity_name_col="member_name", period_start=ps, period_end=pe, output_unit="count", source_batch_id=source_batch_id, contract_version=contract_version)
    append_metric_rows(rows, party_question, metric_id="party_question_count", metric_version=1, value_col="question_count", numerator_col="question_count", denominator_col=None, grain="party", entity_id_col="party_uri", entity_name_col="party_name", period_start=ps, period_end=pe, output_unit="count", source_batch_id=source_batch_id, contract_version=contract_version)
    append_metric_rows(rows, constituency_question, metric_id="constituency_question_count", metric_version=1, value_col="question_count", numerator_col="question_count", denominator_col=None, grain="constituency", entity_id_col="constituency_uri", entity_name_col="constituency_name", period_start=ps, period_end=pe, output_unit="count", source_batch_id=source_batch_id, contract_version=contract_version)

    question_types["entity_id"] = "eligible_tds"
    question_types["entity_name"] = "Eligible TDs"
    append_metric_rows(rows, question_types, metric_id="question_type_share", metric_version=1, value_col="question_type_share", numerator_col="question_count", denominator_col=None, grain="national", entity_id_col="entity_id", entity_name_col="entity_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version, dimension_name="question_type", dimension_value_col="question_type")
    recipients["entity_id"] = "eligible_tds"
    recipients["entity_name"] = "Eligible TDs"
    append_metric_rows(rows, recipients, metric_id="question_recipient_share", metric_version=1, value_col="question_share", numerator_col="question_count", denominator_col=None, grain="national", entity_id_col="entity_id", entity_name_col="entity_name", period_start=ps, period_end=pe, output_unit="proportion", source_batch_id=source_batch_id, contract_version=contract_version, dimension_name="question_recipient", dimension_value_col="to_minister_or_department")

    return metric_results_frame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = load_materialization_contract(CONTRACT_PATH)
    contract_version = int(contract["contract_version"])
    s3 = boto3.client("s3")
    pointer = read_json_required(s3, bucket=BUCKET, key=PRODUCTION_POINTER_KEY)
    source_batch_id = str(pointer.get("batch_id") or pointer.get("mode") or "unknown")

    frames: dict[str, pd.DataFrame] = {}
    resolved_keys: dict[str, str] = {}
    for table, key in TABLE_KEYS.items():
        frames[table], resolved_keys[table] = _read_csv(s3, key)
    frames["speeches"] = canonical_speeches(frames["speeches"])

    period = resolve_period(PERIOD_SPEC)
    foundations = {
        "daily_activity_components": build_daily_activity_components(
            speeches=frames["speeches"], labels=frames["labels"], memberships=frames["memberships"],
            member_parties=frames["parties"], member_constituencies=frames["constituencies"],
            debate_records=frames["debates"], divisions=frames["divisions"], member_votes=frames["votes"],
            questions=frames["questions"], period=period, source_batch_id=source_batch_id,
            contract_version=contract_version,
        ),
        "daily_issue_activity": build_daily_issue_activity(
            speeches=frames["speeches"], labels=frames["labels"], memberships=frames["memberships"],
            member_parties=frames["parties"], member_constituencies=frames["constituencies"],
            period=period, source_batch_id=source_batch_id, contract_version=contract_version,
        ),
        "division_party_vote_components": build_division_party_vote_components(
            frames["votes"], frames["parties"], period=period, source_batch_id=source_batch_id,
            contract_version=contract_version,
        ),
        "daily_question_dimensions": build_daily_question_dimensions(
            questions=frames["questions"], memberships=frames["memberships"], member_parties=frames["parties"],
            member_constituencies=frames["constituencies"], period=period, source_batch_id=source_batch_id,
            contract_version=contract_version,
        ),
    }
    monthly = _monthly_results(
        frames=frames, period=period, source_batch_id=source_batch_id, contract_version=contract_version
    )
    foundations["monthly_metric_results"] = monthly

    manifests = {}
    for name, frame in foundations.items():
        manifests[name] = write_materialized_dataset(
            frame,
            dataset=get_dataset_contract(contract, name),
            output_root=OUT_DIR,
            source_batch_id=source_batch_id,
            contract_version=contract_version,
        )

    report = {
        "commission_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_batch_id": source_batch_id,
        "period": {"start": period.start.isoformat(), "end": period.end.isoformat()},
        "resolved_source_keys": resolved_keys,
        "datasets": {name: {"rows": manifest["row_count"], "files": manifest["files"]} for name, manifest in manifests.items()},
        "monthly_metric_ids": sorted(monthly["metric_id"].unique().tolist()),
        "monthly_metric_row_count": int(len(monthly)),
        "s3_writes_performed": False,
    }
    (OUT_DIR / "commission_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = [
        "# Option A materialization commissioning",
        "",
        f"Source batch: `{source_batch_id}`",
        f"Period: **{period.start.isoformat()} to {period.end.isoformat()}**",
        "",
        "All datasets passed the materialization contract and were written as local workflow artifacts only.",
        "",
    ]
    for name, manifest in manifests.items():
        summary.append(f"- **{name}**: {manifest['row_count']:,} rows")
    summary.extend([
        "",
        f"Monthly result rows: **{len(monthly):,}** across **{monthly['metric_id'].nunique()} metrics**.",
        "",
        "No S3 metric writes or production pointer changes were performed.",
        "",
    ])
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
