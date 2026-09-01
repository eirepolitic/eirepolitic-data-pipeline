from __future__ import annotations

import pandas as pd

from political_metrics.calculators.questions import (
    grouped_question_metrics,
    member_question_metrics,
    prepare_eligible_td_questions,
    question_type_distribution,
    recipient_distribution,
)
from political_metrics.commission import filter_period
from political_metrics.results import append_metric_rows, metric_result_row, metric_results_frame
from process.political_metrics_materialization_commission import build_monthly_results as build_base_monthly_results


MATERIALIZED_METRIC_IDS = {
    "member_speech_count",
    "member_speaking_day_count",
    "member_speeches_per_eligible_debate_day",
    "member_share_of_national_speeches",
    "party_speech_count",
    "constituency_speech_count",
    "national_speech_count",
    "national_speeches_per_debate_day",
    "party_speeches_per_active_member",
    "constituency_speeches_per_active_rep",
    "national_issue_speech_count",
    "national_issue_share",
    "member_issue_share",
    "party_issue_share",
    "party_issue_share_vs_td_national_pp",
    "party_issue_share_vs_average_party_pp",
    "constituency_issue_share",
    "member_vote_participation_pct",
    "party_vote_participation_pct",
    "party_vote_cohesion_pct",
    "constituency_vote_participation_pct",
    "national_division_count",
    "member_question_count",
    "party_question_count",
    "constituency_question_count",
    "question_type_share",
    "question_recipient_count",
    "question_recipient_share",
}


def _name_maps(base: pd.DataFrame) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for grain in ["member", "party", "constituency", "national"]:
        scoped = base[base["grain"].eq(grain)][["entity_id", "entity_name"]].drop_duplicates("entity_id")
        maps[grain] = {
            str(row.entity_id): str(row.entity_name)
            for row in scoped.itertuples(index=False)
            if pd.notna(row.entity_id)
        }
    return maps


def _entity_name(maps: dict[str, dict[str, str]], grain: str, entity_id: object) -> str:
    key = str(entity_id)
    return maps.get(grain, {}).get(key, key)


def _append_question_distribution_rows(
    rows: list[dict],
    *,
    eligible_questions: pd.DataFrame,
    maps: dict[str, dict[str, str]],
    period_start: str,
    period_end: str,
    source_batch_id: str,
    contract_version: int,
) -> None:
    grain_defs = [
        ("national", None, "eligible_tds"),
        ("member", "member_code", None),
        ("party", "party_uri", None),
        ("constituency", "constituency_uri", None),
    ]

    for grain, group_col, fixed_entity in grain_defs:
        qtypes = question_type_distribution(eligible_questions, group_col=group_col)
        if not qtypes.empty:
            if group_col is None:
                typed_total = int(qtypes["question_count"].sum())
                for record in qtypes.to_dict(orient="records"):
                    rows.append(metric_result_row(
                        metric_id="question_type_share",
                        metric_version=1,
                        period_start=period_start,
                        period_end=period_end,
                        grain=grain,
                        entity_id=fixed_entity or "eligible_tds",
                        entity_name="Eligible TDs",
                        dimension_name="question_type",
                        dimension_value=str(record["question_type"]),
                        value=record["question_type_share"],
                        numerator=record["question_count"],
                        denominator=typed_total,
                        output_unit="proportion",
                        source_batch_id=source_batch_id,
                        contract_version=contract_version,
                    ))
            else:
                for record in qtypes.to_dict(orient="records"):
                    entity_id = record[group_col]
                    rows.append(metric_result_row(
                        metric_id="question_type_share",
                        metric_version=1,
                        period_start=period_start,
                        period_end=period_end,
                        grain=grain,
                        entity_id=str(entity_id),
                        entity_name=_entity_name(maps, grain, entity_id),
                        dimension_name="question_type",
                        dimension_value=str(record["question_type"]),
                        value=record["question_type_share"],
                        numerator=record["question_count"],
                        denominator=record["total_question_count"],
                        output_unit="proportion",
                        source_batch_id=source_batch_id,
                        contract_version=contract_version,
                    ))

        recipients = recipient_distribution(eligible_questions, group_col=group_col)
        if not recipients.empty:
            if group_col is None:
                recipient_total = int(recipients["question_count"].sum())
                for record in recipients.to_dict(orient="records"):
                    rows.append(metric_result_row(
                        metric_id="question_recipient_share",
                        metric_version=1,
                        period_start=period_start,
                        period_end=period_end,
                        grain=grain,
                        entity_id=fixed_entity or "eligible_tds",
                        entity_name="Eligible TDs",
                        dimension_name="question_recipient",
                        dimension_value=str(record["to_minister_or_department"]),
                        value=record["question_share"],
                        numerator=record["question_count"],
                        denominator=recipient_total,
                        output_unit="proportion",
                        source_batch_id=source_batch_id,
                        contract_version=contract_version,
                    ))
            else:
                for record in recipients.to_dict(orient="records"):
                    entity_id = record[group_col]
                    rows.append(metric_result_row(
                        metric_id="question_recipient_share",
                        metric_version=1,
                        period_start=period_start,
                        period_end=period_end,
                        grain=grain,
                        entity_id=str(entity_id),
                        entity_name=_entity_name(maps, grain, entity_id),
                        dimension_name="question_recipient",
                        dimension_value=str(record["to_minister_or_department"]),
                        value=record["question_share"],
                        numerator=record["question_count"],
                        denominator=record["total_question_count"],
                        output_unit="proportion",
                        source_batch_id=source_batch_id,
                        contract_version=contract_version,
                    ))


def build_monthly_results(
    *,
    frames: dict[str, pd.DataFrame],
    period,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    """Return the complete catalogue-backed monthly consumer result dataset."""
    base = build_base_monthly_results(
        frames=frames,
        period=period,
        source_batch_id=source_batch_id,
        contract_version=contract_version,
    )
    ps, pe = period.start.isoformat(), period.end.isoformat()
    maps = _name_maps(base)

    # Replace the base national-only question distributions with complete,
    # denominator-backed national/member/party/constituency distributions.
    keep = ~base["metric_id"].isin({"question_type_share", "question_recipient_share"})
    rows = base.loc[keep].to_dict(orient="records")

    member_counts = base[
        base["metric_id"].eq("member_speech_count") & base["grain"].eq("member")
    ].copy()
    td_speech_total = pd.to_numeric(member_counts["value"], errors="coerce").fillna(0).sum()
    if td_speech_total > 0:
        for record in member_counts.to_dict(orient="records"):
            count = float(record["value"])
            rows.append(metric_result_row(
                metric_id="member_share_of_national_speeches",
                metric_version=1,
                period_start=ps,
                period_end=pe,
                grain="member",
                entity_id=str(record["entity_id"]),
                entity_name=str(record["entity_name"]),
                value=count / float(td_speech_total),
                numerator=count,
                denominator=float(td_speech_total),
                output_unit="proportion",
                source_batch_id=source_batch_id,
                contract_version=contract_version,
            ))

    questions = filter_period(frames["questions"], "question_date", period)
    eligible_questions = prepare_eligible_td_questions(
        questions,
        frames["memberships"],
        frames["parties"],
        frames["constituencies"],
    )

    member_question = member_question_metrics(eligible_questions)
    party_question = grouped_question_metrics(eligible_questions, group_col="party_uri")
    constituency_question = grouped_question_metrics(eligible_questions, group_col="constituency_uri")

    for grain, frame, entity_col in [
        ("member", member_question, "member_code"),
        ("party", party_question, "party_uri"),
        ("constituency", constituency_question, "constituency_uri"),
    ]:
        if frame.empty:
            continue
        enriched = frame.copy()
        enriched["entity_name"] = enriched[entity_col].map(lambda value: _entity_name(maps, grain, value))
        append_metric_rows(
            rows,
            enriched,
            metric_id="question_recipient_count",
            metric_version=1,
            value_col="recipient_count",
            numerator_col="recipient_count",
            denominator_col=None,
            grain=grain,
            entity_id_col=entity_col,
            entity_name_col="entity_name",
            period_start=ps,
            period_end=pe,
            output_unit="count",
            source_batch_id=source_batch_id,
            contract_version=contract_version,
        )

    _append_question_distribution_rows(
        rows,
        eligible_questions=eligible_questions,
        maps=maps,
        period_start=ps,
        period_end=pe,
        source_batch_id=source_batch_id,
        contract_version=contract_version,
    )

    return metric_results_frame(rows)
