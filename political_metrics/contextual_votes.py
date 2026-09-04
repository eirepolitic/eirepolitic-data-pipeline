from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from political_metrics.calculators.votes import eligible_division_pairs
from political_metrics.temporal_joins import attach_event_constituency, attach_event_party


DAILY_CONTEXT_VOTE_COLUMNS = [
    "activity_date",
    "division_context",
    "grain",
    "entity_id",
    "component_id",
    "component_value",
    "source_batch_id",
    "component_version",
    "calculated_at_utc",
    "contract_version",
]

CONTEXT_PARTY_VOTE_COLUMNS = [
    "division_id",
    "division_date",
    "division_context",
    "party_uri",
    "vote_code",
    "recorded_vote_count",
    "source_batch_id",
    "component_version",
    "calculated_at_utc",
    "contract_version",
]

ALLOWED_CONTEXTS = {
    "bill_or_legislation",
    "motion_proceeding",
    "procedural_business",
    "other",
}


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in result.columns:
        result[col] = result[col].fillna("").astype(str).str.strip()
    return result


def _stamp(frame: pd.DataFrame, *, source_batch_id: str, contract_version: int) -> pd.DataFrame:
    result = frame.copy()
    result["source_batch_id"] = source_batch_id
    result["component_version"] = 1
    result["calculated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["contract_version"] = contract_version
    return result


def _context_lookup(division_context: pd.DataFrame) -> pd.DataFrame:
    context = _clean(division_context)
    required = {"division_id", "division_context"}
    missing = sorted(required - set(context.columns))
    if missing:
        raise ValueError(f"division_context missing required columns: {missing}")
    if context["division_id"].duplicated().any():
        raise ValueError("division_context contains duplicate division_id values")
    invalid = sorted(set(context["division_context"]) - ALLOWED_CONTEXTS)
    if invalid:
        raise ValueError(f"division_context contains unsupported values: {invalid}")
    return context[["division_id", "division_context"]].copy()


def _filter_period(frame: pd.DataFrame, date_col: str, period) -> pd.DataFrame:
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    return frame.loc[
        dates.between(pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both")
    ].copy()


def _component_rows(
    frame: pd.DataFrame,
    *,
    grain: str,
    entity_col: str,
    component_id: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "activity_date", "division_context", "grain", "entity_id", "component_id", "component_value"
        ])
    data = frame.copy()
    data["activity_date"] = pd.to_datetime(data["division_date"], errors="coerce").dt.date.astype(str)
    data = data[data[entity_col].notna() & data[entity_col].astype(str).ne("")].copy()
    grouped = (
        data.groupby(["activity_date", "division_context", entity_col], dropna=False)
        .size()
        .rename("component_value")
        .reset_index()
        .rename(columns={entity_col: "entity_id"})
    )
    grouped["grain"] = grain
    grouped["component_id"] = component_id
    return grouped[["activity_date", "division_context", "grain", "entity_id", "component_id", "component_value"]]


def build_daily_context_vote_components(
    *,
    divisions: pd.DataFrame,
    member_votes: pd.DataFrame,
    memberships: pd.DataFrame,
    member_parties: pd.DataFrame,
    member_constituencies: pd.DataFrame,
    division_context: pd.DataFrame,
    period,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    context = _context_lookup(division_context)
    period_divisions = _filter_period(divisions, "division_date", period)
    period_divisions = period_divisions.merge(context, on="division_id", how="left", validate="one_to_one")
    if period_divisions["division_context"].isna().any():
        raise ValueError("division_context does not cover every division in the requested period")

    eligible = eligible_division_pairs(memberships, period_divisions)
    eligible = eligible.merge(
        period_divisions[["division_id", "division_context"]],
        on="division_id",
        how="left",
        validate="many_to_one",
    )
    eligible["event_date"] = eligible["division_date"]
    eligible_party = attach_event_party(eligible, member_parties, event_date_col="event_date")
    eligible_const = attach_event_constituency(eligible, member_constituencies, event_date_col="event_date")

    votes = _filter_period(member_votes, "division_date", period)
    votes = votes.merge(context, on="division_id", how="left", validate="many_to_one")
    if votes["division_context"].isna().any():
        raise ValueError("division_context does not cover every member vote in the requested period")
    votes["event_date"] = votes["division_date"]
    votes_party = attach_event_party(votes, member_parties, event_date_col="event_date")
    votes_const = attach_event_constituency(votes, member_constituencies, event_date_col="event_date")

    rows = [
        _component_rows(eligible, grain="member", entity_col="member_code", component_id="eligible_member_division_count"),
        _component_rows(eligible_party, grain="party", entity_col="party_uri", component_id="eligible_member_division_count"),
        _component_rows(eligible_const, grain="constituency", entity_col="constituency_uri", component_id="eligible_member_division_count"),
        _component_rows(votes, grain="member", entity_col="member_code", component_id="recorded_vote_count"),
        _component_rows(votes_party, grain="party", entity_col="party_uri", component_id="recorded_vote_count"),
        _component_rows(votes_const, grain="constituency", entity_col="constituency_uri", component_id="recorded_vote_count"),
    ]
    combined = pd.concat(rows, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=DAILY_CONTEXT_VOTE_COLUMNS)
    combined = (
        combined.groupby(
            ["activity_date", "division_context", "grain", "entity_id", "component_id"],
            as_index=False,
        )["component_value"]
        .sum()
    )
    return _stamp(combined, source_batch_id=source_batch_id, contract_version=contract_version)[DAILY_CONTEXT_VOTE_COLUMNS]


def build_context_division_party_vote_components(
    *,
    member_votes: pd.DataFrame,
    member_parties: pd.DataFrame,
    division_context: pd.DataFrame,
    period,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    context = _context_lookup(division_context)
    votes = _filter_period(member_votes, "division_date", period)
    if votes.empty:
        return pd.DataFrame(columns=CONTEXT_PARTY_VOTE_COLUMNS)
    votes = votes.merge(context, on="division_id", how="left", validate="many_to_one")
    if votes["division_context"].isna().any():
        raise ValueError("division_context does not cover every member vote in the requested period")
    votes["event_date"] = votes["division_date"]
    votes = attach_event_party(votes, member_parties, event_date_col="event_date")
    votes = votes[votes["party_uri"].notna()].copy()
    grouped = (
        votes.groupby(
            ["division_id", "division_date", "division_context", "party_uri", "vote_code"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "recorded_vote_count"})
    )
    return _stamp(grouped, source_batch_id=source_batch_id, contract_version=contract_version)[CONTEXT_PARTY_VOTE_COLUMNS]


def audit_context_vote_reconciliation(
    *,
    daily_context_vote_components: pd.DataFrame,
    context_division_party_vote_components: pd.DataFrame,
    daily_activity_components: pd.DataFrame,
    division_party_vote_components: pd.DataFrame,
) -> dict:
    contextual_daily = _clean(daily_context_vote_components)
    existing_daily = _clean(daily_activity_components)
    contextual_party = _clean(context_division_party_vote_components)
    existing_party = _clean(division_party_vote_components)

    component_ids = {"eligible_member_division_count", "recorded_vote_count"}
    existing_vote_daily = existing_daily[
        existing_daily["component_id"].isin(component_ids)
        & existing_daily["grain"].isin({"member", "party", "constituency"})
    ].copy()

    daily_collapsed = (
        contextual_daily.groupby(["activity_date", "grain", "entity_id", "component_id"], as_index=False)["component_value"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
        .rename(columns={"component_value": "context_value"})
    )
    existing_vote_daily["existing_value"] = pd.to_numeric(existing_vote_daily["component_value"], errors="coerce").fillna(0)
    daily_compare = existing_vote_daily[["activity_date", "grain", "entity_id", "component_id", "existing_value"]].merge(
        daily_collapsed,
        on=["activity_date", "grain", "entity_id", "component_id"],
        how="outer",
    ).fillna(0)
    daily_mismatch = int((daily_compare["existing_value"] != daily_compare["context_value"]).sum())

    party_collapsed = (
        contextual_party.groupby(["division_id", "division_date", "party_uri", "vote_code"], as_index=False)["recorded_vote_count"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
        .rename(columns={"recorded_vote_count": "context_value"})
    )
    existing_party["existing_value"] = pd.to_numeric(existing_party["recorded_vote_count"], errors="coerce").fillna(0)
    party_compare = existing_party[["division_id", "division_date", "party_uri", "vote_code", "existing_value"]].merge(
        party_collapsed,
        on=["division_id", "division_date", "party_uri", "vote_code"],
        how="outer",
    ).fillna(0)
    party_mismatch = int((party_compare["existing_value"] != party_compare["context_value"]).sum())

    invalid_context_daily = int((~contextual_daily["division_context"].isin(ALLOWED_CONTEXTS)).sum())
    invalid_context_party = int((~contextual_party["division_context"].isin(ALLOWED_CONTEXTS)).sum())
    daily_duplicate_keys = int(contextual_daily.duplicated([
        "activity_date", "division_context", "grain", "entity_id", "component_id"
    ]).sum())
    party_duplicate_keys = int(contextual_party.duplicated([
        "division_id", "division_context", "party_uri", "vote_code"
    ]).sum())

    checks = {
        "daily_context_values_allowed": invalid_context_daily == 0,
        "party_context_values_allowed": invalid_context_party == 0,
        "daily_primary_key_unique": daily_duplicate_keys == 0,
        "party_primary_key_unique": party_duplicate_keys == 0,
        "daily_components_reconcile_to_existing": daily_mismatch == 0,
        "party_components_reconcile_to_existing": party_mismatch == 0,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "daily_row_count": int(len(contextual_daily)),
        "party_row_count": int(len(contextual_party)),
        "daily_reconciliation_mismatches": daily_mismatch,
        "party_reconciliation_mismatches": party_mismatch,
        "daily_duplicate_keys": daily_duplicate_keys,
        "party_duplicate_keys": party_duplicate_keys,
        "daily_context_counts": {k: int(v) for k, v in contextual_daily["division_context"].value_counts().to_dict().items()},
        "party_context_counts": {k: int(v) for k, v in contextual_party["division_context"].value_counts().to_dict().items()},
    }
