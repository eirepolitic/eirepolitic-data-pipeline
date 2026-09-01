from __future__ import annotations

import pandas as pd

from .calculators.speeches import grouped_speech_metrics, member_speech_metrics, national_speech_metrics
from .eligibility import constituency_debate_day_exposure, member_debate_day_exposure, party_debate_day_exposure
from .periods import MetricPeriod
from .temporal_joins import attach_event_constituency, attach_event_membership, attach_event_party


def filter_period(frame: pd.DataFrame, date_col: str, period: MetricPeriod) -> pd.DataFrame:
    data = frame.copy()
    dates = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    return data.loc[dates.between(start, end, inclusive="both")].copy()


def prepare_eligible_td_speeches(
    speeches: pd.DataFrame,
    memberships: pd.DataFrame,
    member_parties: pd.DataFrame,
    member_constituencies: pd.DataFrame,
    period: MetricPeriod,
) -> pd.DataFrame:
    """Prepare period-correct TD speech facts for member/party/constituency metrics."""
    period_speeches = filter_period(speeches, "debate_date", period)
    identified = period_speeches[period_speeches["member_code"].notna()].copy()
    if identified.empty:
        return identified.assign(party_uri=pd.Series(dtype="object"), constituency_uri=pd.Series(dtype="object"))

    identified["event_date"] = identified["debate_date"]
    known_codes = set(memberships["member_code"].dropna().astype(str))
    identified = identified[identified["member_code"].astype(str).isin(known_codes)].copy()
    if identified.empty:
        return identified.assign(party_uri=pd.Series(dtype="object"), constituency_uri=pd.Series(dtype="object"))

    joined = attach_event_membership(identified, memberships, event_date_col="event_date")
    joined = joined[joined["membership_id"].notna()].copy()
    if "chamber" in joined.columns:
        joined = joined[joined["chamber"].fillna("").str.lower().eq("dail")].copy()
    if joined.empty:
        return joined.assign(party_uri=pd.Series(dtype="object"), constituency_uri=pd.Series(dtype="object"))

    joined = attach_event_party(joined, member_parties, event_date_col="event_date")
    joined = attach_event_constituency(joined, member_constituencies, event_date_col="event_date")
    return joined


def calculate_core_speech_metrics(
    *,
    speeches: pd.DataFrame,
    memberships: pd.DataFrame,
    member_parties: pd.DataFrame,
    member_constituencies: pd.DataFrame,
    debate_records: pd.DataFrame,
    period: MetricPeriod,
) -> dict[str, pd.DataFrame | dict]:
    """Calculate the first commissioned speech measures without materializing them."""
    period_speeches = filter_period(speeches, "debate_date", period)
    period_debates = filter_period(debate_records, "debate_date", period)
    debate_days = (
        pd.to_datetime(period_debates["debate_date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    eligible = prepare_eligible_td_speeches(
        speeches,
        memberships,
        member_parties,
        member_constituencies,
        period,
    )

    member_exposure = member_debate_day_exposure(memberships, debate_days)
    member_exposure = member_exposure[member_exposure["eligible_debate_days"] > 0].copy()
    party_exposure = party_debate_day_exposure(member_parties, debate_days)
    constituency_exposure = constituency_debate_day_exposure(member_constituencies, debate_days)

    member = member_speech_metrics(eligible, member_exposure)
    party = grouped_speech_metrics(eligible, group_col="party_uri", group_exposure=party_exposure)
    constituency = grouped_speech_metrics(
        eligible,
        group_col="constituency_uri",
        group_exposure=constituency_exposure,
    )
    national = national_speech_metrics(period_speeches)

    return {
        "period_speeches": period_speeches,
        "eligible_td_speeches": eligible,
        "debate_days": pd.DataFrame({"debate_date": debate_days}),
        "member_exposure": member_exposure,
        "party_exposure": party_exposure,
        "constituency_exposure": constituency_exposure,
        "member_metrics": member,
        "party_metrics": party,
        "constituency_metrics": constituency,
        "national_metrics": national,
    }


def reconciliation_checks(results: dict[str, pd.DataFrame | dict]) -> dict[str, bool | int]:
    period_speeches = results["period_speeches"]
    eligible = results["eligible_td_speeches"]
    member = results["member_metrics"]
    party = results["party_metrics"]
    constituency = results["constituency_metrics"]
    national = results["national_metrics"]

    assert isinstance(period_speeches, pd.DataFrame)
    assert isinstance(eligible, pd.DataFrame)
    assert isinstance(member, pd.DataFrame)
    assert isinstance(party, pd.DataFrame)
    assert isinstance(constituency, pd.DataFrame)
    assert isinstance(national, dict)

    national_count = int(period_speeches["speech_id"].dropna().nunique())
    eligible_count = int(eligible["speech_id"].dropna().nunique())
    member_sum = int(member["speech_count"].sum()) if not member.empty else 0
    party_sum = int(party["speech_count"].sum()) if not party.empty else 0
    constituency_sum = int(constituency["speech_count"].sum()) if not constituency.empty else 0

    return {
        "national_distinct_speeches": national_count,
        "eligible_td_distinct_speeches": eligible_count,
        "member_speech_sum": member_sum,
        "party_speech_sum": party_sum,
        "constituency_speech_sum": constituency_sum,
        "national_calculator_matches": int(national.get("speech_count", -1)) == national_count,
        "member_sum_matches_eligible_td_speeches": member_sum == eligible_count,
        "party_sum_matches_eligible_td_speeches": party_sum == eligible_count,
        "constituency_sum_matches_eligible_td_speeches": constituency_sum == eligible_count,
    }
