from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


CONTEXTS = ["bill_or_legislation", "motion_proceeding", "procedural_business", "other"]
RESULT_COLUMNS = [
    "metric_id","metric_version","period_type","period_start","period_end","grain","entity_id","entity_name",
    "dimension_name","dimension_value","value","numerator","denominator","output_unit","reliability_status",
    "public_use_status","warning_code","source_batch_id","calculated_at_utc","contract_version",
]


def member_context_reliability(denominator: int) -> tuple[str, str, str]:
    if denominator >= 25:
        return "reliable", "suitable", "none"
    if denominator >= 10:
        return "caution", "suitable_with_context", "small_context_sample"
    if denominator >= 5:
        return "caution", "suitable_with_context", "small_context_sample"
    return "insufficient_for_comparison", "not_certified", "insufficient_context_sample"


def party_context_reliability(qualifying_divisions: int) -> tuple[str, str, str]:
    if qualifying_divisions >= 10:
        return "reliable", "suitable_with_context", "none"
    if qualifying_divisions >= 5:
        return "caution", "suitable_with_context", "small_division_sample"
    return "insufficient_for_comparison", "not_certified", "insufficient_division_sample"


def _period_filter(frame: pd.DataFrame, date_col: str, period) -> pd.DataFrame:
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    return frame.loc[dates.between(pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both")].copy()


def build_monthly_contextual_vote_results(
    *,
    daily_context_vote_components: pd.DataFrame,
    context_division_party_vote_components: pd.DataFrame,
    period,
    source_batch_id: str,
    contract_version: int,
    metric_version: int = 1,
) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    daily = _period_filter(daily_context_vote_components, "activity_date", period)
    daily["component_value"] = pd.to_numeric(daily["component_value"], errors="coerce").fillna(0)
    rows: list[dict] = []

    member = daily[daily["grain"].eq("member")].copy()
    if not member.empty:
        pivot = member.pivot_table(
            index=["division_context", "entity_id"],
            columns="component_id",
            values="component_value",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        for col in ["recorded_vote_count", "eligible_member_division_count"]:
            if col not in pivot.columns:
                pivot[col] = 0
        for r in pivot.itertuples(index=False):
            numerator = int(r.recorded_vote_count)
            denominator = int(r.eligible_member_division_count)
            value = (numerator / denominator) if denominator else None
            reliability, public_use, warning = member_context_reliability(denominator)
            rows.append({
                "metric_id": "member_vote_participation_pct", "metric_version": metric_version,
                "period_type": "calendar_month", "period_start": period.start.isoformat(), "period_end": period.end.isoformat(),
                "grain": "member", "entity_id": r.entity_id, "entity_name": r.entity_id,
                "dimension_name": "division_context", "dimension_value": r.division_context,
                "value": value, "numerator": numerator, "denominator": denominator, "output_unit": "proportion",
                "reliability_status": reliability, "public_use_status": public_use, "warning_code": warning,
                "source_batch_id": source_batch_id, "calculated_at_utc": now, "contract_version": contract_version,
            })

    party = _period_filter(context_division_party_vote_components, "division_date", period)
    party["recorded_vote_count"] = pd.to_numeric(party["recorded_vote_count"], errors="coerce").fillna(0)
    if not party.empty:
        div = party.groupby(["division_context", "party_uri", "division_id", "vote_code"], as_index=False)["recorded_vote_count"].sum()
        totals = div.groupby(["division_context", "party_uri", "division_id"], as_index=False)["recorded_vote_count"].sum().rename(columns={"recorded_vote_count":"division_total"})
        modal = div.groupby(["division_context", "party_uri", "division_id"], as_index=False)["recorded_vote_count"].max().rename(columns={"recorded_vote_count":"aligned_votes"})
        qual = totals.merge(modal, on=["division_context", "party_uri", "division_id"], how="inner")
        qual = qual[qual["division_total"] >= 2].copy()
        agg = qual.groupby(["division_context", "party_uri"], as_index=False).agg(
            qualifying_divisions=("division_id", "nunique"), aligned_votes=("aligned_votes", "sum"), total_votes=("division_total", "sum")
        )
        for r in agg.itertuples(index=False):
            numerator = int(r.aligned_votes)
            denominator = int(r.total_votes)
            value = (numerator / denominator) if denominator else None
            reliability, public_use, warning = party_context_reliability(int(r.qualifying_divisions))
            if "independent" in str(r.party_uri).lower():
                warning = "independent_group_agreement" if warning == "none" else f"{warning};independent_group_agreement"
                public_use = "suitable_with_context" if public_use == "suitable" else public_use
            rows.append({
                "metric_id": "party_vote_cohesion_pct", "metric_version": metric_version,
                "period_type": "calendar_month", "period_start": period.start.isoformat(), "period_end": period.end.isoformat(),
                "grain": "party", "entity_id": r.party_uri, "entity_name": r.party_uri,
                "dimension_name": "division_context", "dimension_value": r.division_context,
                "value": value, "numerator": numerator, "denominator": denominator, "output_unit": "proportion",
                "reliability_status": reliability, "public_use_status": public_use, "warning_code": warning,
                "source_batch_id": source_batch_id, "calculated_at_utc": now, "contract_version": contract_version,
            })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def audit_monthly_contextual_vote_results(*, results: pd.DataFrame, periods: list, source_batch_id: str) -> dict:
    frame = results.copy()
    duplicate = int(frame.duplicated([
        "metric_id","metric_version","period_start","period_end","grain","entity_id","dimension_name","dimension_value"
    ]).sum())
    invalid_dimensions = int((frame["dimension_name"] != "division_context").sum()) if not frame.empty else 0
    invalid_contexts = int((~frame["dimension_value"].isin(CONTEXTS)).sum()) if not frame.empty else 0
    wrong_batch = int((frame["source_batch_id"] != source_batch_id).sum()) if not frame.empty else 0
    invalid_metric = int((~frame["metric_id"].isin({"member_vote_participation_pct","party_vote_cohesion_pct"})).sum()) if not frame.empty else 0
    values = pd.to_numeric(frame["value"], errors="coerce") if not frame.empty else pd.Series(dtype=float)
    value_out_of_range = int(((values < 0) | (values > 1)).sum()) if not frame.empty else 0
    expected_periods = {(p.start.isoformat(), p.end.isoformat()) for p in periods}
    actual_periods = set(zip(frame["period_start"], frame["period_end"])) if not frame.empty else set()
    unexpected_periods = len(actual_periods - expected_periods)
    checks = {
        "primary_key_unique": duplicate == 0, "dimension_name_valid": invalid_dimensions == 0,
        "context_values_valid": invalid_contexts == 0, "source_batch_consistent": wrong_batch == 0,
        "metric_ids_valid": invalid_metric == 0, "proportion_values_in_range": value_out_of_range == 0,
        "periods_valid": unexpected_periods == 0,
    }
    return {
        "ready": all(checks.values()), "checks": checks, "row_count": int(len(frame)),
        "metric_counts": {k:int(v) for k,v in frame["metric_id"].value_counts().to_dict().items()},
        "context_counts": {k:int(v) for k,v in frame["dimension_value"].value_counts().to_dict().items()},
        "reliability_counts": {k:int(v) for k,v in frame["reliability_status"].value_counts().to_dict().items()},
        "duplicate_rows": duplicate, "invalid_contexts": invalid_contexts,
    }
