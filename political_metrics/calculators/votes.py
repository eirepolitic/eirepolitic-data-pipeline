from __future__ import annotations

import pandas as pd

from political_metrics.eligibility import eligible_member_events
from political_metrics.temporal_joins import attach_event_constituency, attach_event_party


def eligible_division_pairs(memberships: pd.DataFrame, divisions: pd.DataFrame) -> pd.DataFrame:
    return eligible_member_events(
        memberships,
        divisions,
        event_id_col="division_id",
        event_date_col="division_date",
    )


def member_vote_participation(
    member_votes: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Recorded vote participation among divisions for which each TD was eligible."""
    eligible = (
        eligible_pairs.groupby("member_code")["division_id"]
        .nunique()
        .rename("eligible_division_count")
        .reset_index()
    )
    cast = (
        member_votes.groupby("member_code")["division_id"]
        .nunique()
        .rename("votes_cast_count")
        .reset_index()
    )
    result = eligible.merge(cast, on="member_code", how="left")
    result["votes_cast_count"] = result["votes_cast_count"].fillna(0).astype(int)
    result["vote_participation_pct"] = result["votes_cast_count"].div(
        result["eligible_division_count"].replace(0, pd.NA)
    ).astype("Float64")
    return result


def vote_unity_reliability(qualifying_divisions: int) -> str:
    """Reliability label used by public voting-unity comparisons."""
    if qualifying_divisions >= 10:
        return "reliable"
    if qualifying_divisions >= 5:
        return "caution"
    return "insufficient_for_comparison"


def party_vote_metrics(
    member_votes: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    member_parties: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate party vote participation and recorded voting unity.

    Unity is the weighted share of recorded party votes that match the most
    common vote within that party for each division. Divisions with only one
    recorded party voter are excluded from the unity denominator because they do
    not demonstrate within-party agreement.
    """
    eligible = eligible_pairs.copy()
    eligible["event_date"] = eligible["division_date"]
    eligible = attach_event_party(eligible, member_parties, event_date_col="event_date")
    eligible = eligible[eligible["party_uri"].notna()].copy()

    votes = member_votes.copy()
    votes["event_date"] = votes["division_date"]
    votes = attach_event_party(votes, member_parties, event_date_col="event_date")
    votes = votes[votes["party_uri"].notna()].copy()

    eligible_counts = (
        eligible.groupby("party_uri")
        .size()
        .rename("eligible_member_divisions")
        .reset_index()
    )
    cast_counts = votes.groupby("party_uri").size().rename("recorded_member_votes").reset_index()

    result = eligible_counts.merge(cast_counts, on="party_uri", how="left")
    result["recorded_member_votes"] = result["recorded_member_votes"].fillna(0).astype(int)
    result["vote_participation_pct"] = result["recorded_member_votes"].div(
        result["eligible_member_divisions"].replace(0, pd.NA)
    ).astype("Float64")

    if votes.empty:
        result["qualifying_unity_divisions"] = 0
        result["unity_votes_aligned"] = 0
        result["unity_votes_total"] = 0
        result["vote_cohesion_pct"] = pd.NA
        result["unity_reliability_status"] = "insufficient_for_comparison"
        result["unity_public_safe"] = False
        return result

    counts = (
        votes.groupby(["party_uri", "division_id", "vote_code"])
        .size()
        .rename("vote_count")
        .reset_index()
    )
    division_totals = (
        counts.groupby(["party_uri", "division_id"])["vote_count"]
        .agg([("participating_votes", "sum"), ("modal_votes", "max")])
        .reset_index()
    )
    qualifying = division_totals[division_totals["participating_votes"] >= 2].copy()
    unity = (
        qualifying.groupby("party_uri")
        .agg(
            qualifying_unity_divisions=("division_id", "nunique"),
            unity_votes_aligned=("modal_votes", "sum"),
            unity_votes_total=("participating_votes", "sum"),
        )
        .reset_index()
    )
    result = result.merge(unity, on="party_uri", how="left")
    for col in ["qualifying_unity_divisions", "unity_votes_aligned", "unity_votes_total"]:
        result[col] = result[col].fillna(0).astype(int)
    result["vote_cohesion_pct"] = result["unity_votes_aligned"].div(
        result["unity_votes_total"].replace(0, pd.NA)
    ).astype("Float64")
    result["unity_reliability_status"] = result["qualifying_unity_divisions"].map(
        lambda value: vote_unity_reliability(int(value))
    )
    result["unity_public_safe"] = result["unity_reliability_status"].eq("reliable")
    return result


def constituency_vote_participation(
    member_votes: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    member_constituencies: pd.DataFrame,
) -> pd.DataFrame:
    eligible = eligible_pairs.copy()
    eligible["event_date"] = eligible["division_date"]
    eligible = attach_event_constituency(eligible, member_constituencies, event_date_col="event_date")
    eligible = eligible[eligible["constituency_uri"].notna()].copy()

    votes = member_votes.copy()
    votes["event_date"] = votes["division_date"]
    votes = attach_event_constituency(votes, member_constituencies, event_date_col="event_date")
    votes = votes[votes["constituency_uri"].notna()].copy()

    eligible_counts = eligible.groupby("constituency_uri").size().rename("eligible_member_divisions").reset_index()
    cast_counts = votes.groupby("constituency_uri").size().rename("recorded_member_votes").reset_index()
    result = eligible_counts.merge(cast_counts, on="constituency_uri", how="left")
    result["recorded_member_votes"] = result["recorded_member_votes"].fillna(0).astype(int)
    result["vote_participation_pct"] = result["recorded_member_votes"].div(
        result["eligible_member_divisions"].replace(0, pd.NA)
    ).astype("Float64")
    return result
