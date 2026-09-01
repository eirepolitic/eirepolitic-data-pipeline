from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from political_metrics.calculators.issues import attach_issue_labels, policy_speeches
from political_metrics.calculators.questions import prepare_eligible_td_questions
from political_metrics.calculators.votes import eligible_division_pairs
from political_metrics.commission import prepare_eligible_td_speeches
from political_metrics.temporal_joins import attach_event_constituency, attach_event_party


ACTIVITY_COLUMNS = [
    "activity_date", "grain", "entity_id", "component_id", "component_value",
    "source_batch_id", "component_version", "calculated_at_utc", "contract_version",
]
ISSUE_COLUMNS = [
    "activity_date", "grain", "entity_id", "issue_label", "issue_speech_count",
    "policy_speech_count", "source_batch_id", "component_version", "calculated_at_utc",
    "contract_version",
]
VOTE_COLUMNS = [
    "division_id", "division_date", "party_uri", "vote_code", "recorded_vote_count",
    "source_batch_id", "component_version", "calculated_at_utc", "contract_version",
]
QUESTION_DIMENSION_COLUMNS = [
    "activity_date", "grain", "entity_id", "dimension_name", "dimension_value",
    "question_count", "source_batch_id", "component_version", "calculated_at_utc",
    "contract_version",
]


def _stamp(frame: pd.DataFrame, *, source_batch_id: str, contract_version: int, component_version: int = 1) -> pd.DataFrame:
    result = frame.copy()
    result["source_batch_id"] = source_batch_id
    result["component_version"] = component_version
    result["calculated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["contract_version"] = contract_version
    return result


def _activity_rows(
    frame: pd.DataFrame,
    *,
    date_col: str,
    grain: str,
    entity_col: str | None,
    component_id: str,
    value_col: str | None = None,
    count_distinct_col: str | None = None,
) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        return pd.DataFrame(columns=["activity_date", "grain", "entity_id", "component_id", "component_value"])
    data["activity_date"] = pd.to_datetime(data[date_col], errors="coerce").dt.date.astype(str)
    data["entity_id"] = "national" if entity_col is None else data[entity_col].astype(str)
    keys = ["activity_date", "entity_id"]
    if value_col is not None:
        grouped = data.groupby(keys, dropna=False)[value_col].sum().rename("component_value").reset_index()
    elif count_distinct_col is not None:
        grouped = data.groupby(keys, dropna=False)[count_distinct_col].nunique().rename("component_value").reset_index()
    else:
        grouped = data.groupby(keys, dropna=False).size().rename("component_value").reset_index()
    grouped["grain"] = grain
    grouped["component_id"] = component_id
    return grouped[["activity_date", "grain", "entity_id", "component_id", "component_value"]]


def build_daily_activity_components(
    *,
    speeches: pd.DataFrame,
    labels: pd.DataFrame,
    memberships: pd.DataFrame,
    member_parties: pd.DataFrame,
    member_constituencies: pd.DataFrame,
    debate_records: pd.DataFrame,
    divisions: pd.DataFrame,
    member_votes: pd.DataFrame,
    questions: pd.DataFrame,
    period,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    eligible_speeches = prepare_eligible_td_speeches(
        speeches, memberships, member_parties, member_constituencies, period
    )
    eligible_speeches = attach_issue_labels(eligible_speeches, labels)
    policy = policy_speeches(eligible_speeches)

    period_speeches = speeches.copy()
    speech_dates = pd.to_datetime(period_speeches["debate_date"], errors="coerce")
    period_speeches = period_speeches.loc[
        speech_dates.between(pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both")
    ].copy()

    period_debates = debate_records.copy()
    debate_dates = pd.to_datetime(period_debates["debate_date"], errors="coerce")
    period_debates = period_debates.loc[
        debate_dates.between(pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both")
    ].copy()
    debate_days = (
        period_debates[["debate_date"]]
        .dropna()
        .drop_duplicates()
        .assign(debate_day_count=1)
    )

    period_divisions = divisions.copy()
    division_dates = pd.to_datetime(period_divisions["division_date"], errors="coerce")
    period_divisions = period_divisions.loc[
        division_dates.between(pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both")
    ].copy()
    eligible_pairs = eligible_division_pairs(memberships, period_divisions)
    eligible_pairs["event_date"] = eligible_pairs["division_date"]
    eligible_party = attach_event_party(eligible_pairs, member_parties, event_date_col="event_date")
    eligible_const = attach_event_constituency(eligible_pairs, member_constituencies, event_date_col="event_date")

    period_votes = member_votes.copy()
    vote_dates = pd.to_datetime(period_votes["division_date"], errors="coerce")
    period_votes = period_votes.loc[
        vote_dates.between(pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both")
    ].copy()
    period_votes["event_date"] = period_votes["division_date"]
    votes_party = attach_event_party(period_votes, member_parties, event_date_col="event_date")
    votes_const = attach_event_constituency(period_votes, member_constituencies, event_date_col="event_date")

    period_questions = questions.copy()
    qdates = pd.to_datetime(period_questions["question_date"], errors="coerce")
    period_questions = period_questions.loc[
        qdates.between(pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both")
    ].copy()
    eligible_questions = prepare_eligible_td_questions(
        period_questions, memberships, member_parties, member_constituencies
    )

    rows: list[pd.DataFrame] = []

    rows.append(_activity_rows(period_speeches, date_col="debate_date", grain="national", entity_col=None, component_id="speech_count", count_distinct_col="speech_id"))
    for grain, col in [("member", "member_code"), ("party", "party_uri"), ("constituency", "constituency_uri")]:
        rows.append(_activity_rows(eligible_speeches, date_col="debate_date", grain=grain, entity_col=col, component_id="speech_count", count_distinct_col="speech_id"))
        rows.append(_activity_rows(policy, date_col="debate_date", grain=grain, entity_col=col, component_id="policy_speech_count", count_distinct_col="speech_id"))

    rows.append(_activity_rows(policy, date_col="debate_date", grain="national", entity_col=None, component_id="policy_speech_count", count_distinct_col="speech_id"))

    member_speaking = eligible_speeches[["debate_date", "member_code"]].drop_duplicates().assign(value=1)
    rows.append(_activity_rows(member_speaking, date_col="debate_date", grain="member", entity_col="member_code", component_id="speaking_day_count", value_col="value"))

    rows.append(_activity_rows(debate_days, date_col="debate_date", grain="national", entity_col=None, component_id="debate_day_count", value_col="debate_day_count"))

    if not debate_days.empty:
        debate_event = debate_days.rename(columns={"debate_date": "event_date"})[["event_date"]].copy()
        debate_event["debate_day_id"] = pd.to_datetime(debate_event["event_date"]).dt.date.astype(str)
        from political_metrics.eligibility import eligible_member_events
        eligible_debate = eligible_member_events(
            memberships,
            debate_event.rename(columns={"event_date": "debate_date"}),
            event_id_col="debate_day_id",
            event_date_col="debate_date",
        ).assign(value=1)
        rows.append(_activity_rows(eligible_debate, date_col="debate_date", grain="member", entity_col="member_code", component_id="member_debate_day_exposure", value_col="value"))
        eligible_debate["event_date"] = eligible_debate["debate_date"]
        party_debate = attach_event_party(eligible_debate, member_parties, event_date_col="event_date")
        const_debate = attach_event_constituency(eligible_debate, member_constituencies, event_date_col="event_date")
        rows.append(_activity_rows(party_debate, date_col="debate_date", grain="party", entity_col="party_uri", component_id="member_debate_day_exposure", value_col="value"))
        rows.append(_activity_rows(const_debate, date_col="debate_date", grain="constituency", entity_col="constituency_uri", component_id="member_debate_day_exposure", value_col="value"))

    rows.append(_activity_rows(period_divisions, date_col="division_date", grain="national", entity_col=None, component_id="division_count", count_distinct_col="division_id"))
    rows.append(_activity_rows(eligible_pairs.assign(value=1), date_col="division_date", grain="member", entity_col="member_code", component_id="eligible_member_division_count", value_col="value"))
    rows.append(_activity_rows(eligible_party.assign(value=1), date_col="division_date", grain="party", entity_col="party_uri", component_id="eligible_member_division_count", value_col="value"))
    rows.append(_activity_rows(eligible_const.assign(value=1), date_col="division_date", grain="constituency", entity_col="constituency_uri", component_id="eligible_member_division_count", value_col="value"))
    rows.append(_activity_rows(period_votes, date_col="division_date", grain="member", entity_col="member_code", component_id="recorded_vote_count"))
    rows.append(_activity_rows(votes_party, date_col="division_date", grain="party", entity_col="party_uri", component_id="recorded_vote_count"))
    rows.append(_activity_rows(votes_const, date_col="division_date", grain="constituency", entity_col="constituency_uri", component_id="recorded_vote_count"))

    for grain, col in [("member", "member_code"), ("party", "party_uri"), ("constituency", "constituency_uri")]:
        rows.append(_activity_rows(eligible_questions, date_col="question_date", grain=grain, entity_col=col, component_id="question_count", count_distinct_col="question_id"))
    rows.append(_activity_rows(eligible_questions, date_col="question_date", grain="national", entity_col=None, component_id="question_count", count_distinct_col="question_id"))

    combined = pd.concat(rows, ignore_index=True)
    combined = combined[combined["entity_id"].notna() & combined["entity_id"].astype(str).ne("nan")].copy()
    combined = (
        combined.groupby(["activity_date", "grain", "entity_id", "component_id"], as_index=False)["component_value"]
        .sum()
    )
    return _stamp(combined, source_batch_id=source_batch_id, contract_version=contract_version)[ACTIVITY_COLUMNS]


def build_daily_issue_activity(
    *,
    speeches: pd.DataFrame,
    labels: pd.DataFrame,
    memberships: pd.DataFrame,
    member_parties: pd.DataFrame,
    member_constituencies: pd.DataFrame,
    period,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    eligible = prepare_eligible_td_speeches(speeches, memberships, member_parties, member_constituencies, period)
    eligible = attach_issue_labels(eligible, labels)
    policy = policy_speeches(eligible)
    if policy.empty:
        return pd.DataFrame(columns=ISSUE_COLUMNS)

    policy["activity_date"] = pd.to_datetime(policy["debate_date"], errors="coerce").dt.date.astype(str)
    outputs: list[pd.DataFrame] = []
    for grain, col in [("member", "member_code"), ("party", "party_uri"), ("constituency", "constituency_uri")]:
        data = policy[policy[col].notna()].copy()
        issue = data.groupby(["activity_date", col, "issue_label"])["speech_id"].nunique().rename("issue_speech_count").reset_index()
        total = data.groupby(["activity_date", col])["speech_id"].nunique().rename("policy_speech_count").reset_index()
        merged = issue.merge(total, on=["activity_date", col], how="left").rename(columns={col: "entity_id"})
        merged["grain"] = grain
        outputs.append(merged[["activity_date", "grain", "entity_id", "issue_label", "issue_speech_count", "policy_speech_count"]])

    issue = policy.groupby(["activity_date", "issue_label"])["speech_id"].nunique().rename("issue_speech_count").reset_index()
    total = policy.groupby("activity_date")["speech_id"].nunique().rename("policy_speech_count").reset_index()
    national = issue.merge(total, on="activity_date", how="left")
    national["grain"] = "national"
    national["entity_id"] = "national"
    outputs.append(national[["activity_date", "grain", "entity_id", "issue_label", "issue_speech_count", "policy_speech_count"]])

    combined = pd.concat(outputs, ignore_index=True)
    return _stamp(combined, source_batch_id=source_batch_id, contract_version=contract_version)[ISSUE_COLUMNS]


def build_division_party_vote_components(
    member_votes: pd.DataFrame,
    member_parties: pd.DataFrame,
    *,
    period,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    dates = pd.to_datetime(member_votes["division_date"], errors="coerce")
    votes = member_votes.loc[dates.between(pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both")].copy()
    if votes.empty:
        return pd.DataFrame(columns=VOTE_COLUMNS)
    votes["event_date"] = votes["division_date"]
    votes = attach_event_party(votes, member_parties, event_date_col="event_date")
    votes = votes[votes["party_uri"].notna()].copy()
    grouped = (
        votes.groupby(["division_id", "division_date", "party_uri", "vote_code"], as_index=False)
        .size()
        .rename(columns={"size": "recorded_vote_count"})
    )
    return _stamp(grouped, source_batch_id=source_batch_id, contract_version=contract_version)[VOTE_COLUMNS]


def build_daily_question_dimensions(
    *,
    questions: pd.DataFrame,
    memberships: pd.DataFrame,
    member_parties: pd.DataFrame,
    member_constituencies: pd.DataFrame,
    period,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    dates = pd.to_datetime(questions["question_date"], errors="coerce")
    period_questions = questions.loc[dates.between(pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both")].copy()
    eligible = prepare_eligible_td_questions(period_questions, memberships, member_parties, member_constituencies)
    if eligible.empty:
        return pd.DataFrame(columns=QUESTION_DIMENSION_COLUMNS)
    eligible["activity_date"] = pd.to_datetime(eligible["question_date"], errors="coerce").dt.date.astype(str)

    outputs: list[pd.DataFrame] = []
    dimensions = [("question_type", "question_type"), ("question_recipient", "to_minister_or_department")]
    grains = [("member", "member_code"), ("party", "party_uri"), ("constituency", "constituency_uri")]
    for dimension_name, source_col in dimensions:
        for grain, entity_col in grains:
            data = eligible[eligible[entity_col].notna() & eligible[source_col].notna()].copy()
            data[source_col] = data[source_col].astype(str).str.strip()
            data = data[data[source_col].ne("")]
            grouped = (
                data.groupby(["activity_date", entity_col, source_col])["question_id"]
                .nunique()
                .rename("question_count")
                .reset_index()
                .rename(columns={entity_col: "entity_id", source_col: "dimension_value"})
            )
            grouped["grain"] = grain
            grouped["dimension_name"] = dimension_name
            outputs.append(grouped[["activity_date", "grain", "entity_id", "dimension_name", "dimension_value", "question_count"]])

        data = eligible[eligible[source_col].notna()].copy()
        data[source_col] = data[source_col].astype(str).str.strip()
        data = data[data[source_col].ne("")]
        grouped = (
            data.groupby(["activity_date", source_col])["question_id"]
            .nunique()
            .rename("question_count")
            .reset_index()
            .rename(columns={source_col: "dimension_value"})
        )
        grouped["grain"] = "national"
        grouped["entity_id"] = "national"
        grouped["dimension_name"] = dimension_name
        outputs.append(grouped[["activity_date", "grain", "entity_id", "dimension_name", "dimension_value", "question_count"]])

    combined = pd.concat(outputs, ignore_index=True)
    return _stamp(combined, source_batch_id=source_batch_id, contract_version=contract_version)[QUESTION_DIMENSION_COLUMNS]
