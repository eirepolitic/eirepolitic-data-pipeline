from __future__ import annotations

import pandas as pd

from political_metrics.issue_audit import APPROVED_ISSUES

POLICY_ISSUES = sorted(APPROVED_ISSUES - {"NONE"})


def attach_issue_labels(speeches: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach one final issue label to each speech by stable speech ID."""
    required = {"speech_id", "issue_label", "classification_status", "source_speech_text_hash"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"issue labels missing required columns: {missing}")
    if labels["speech_id"].duplicated().any():
        raise ValueError("issue labels contain duplicate speech_id values")
    merged = speeches.merge(
        labels[["speech_id", "issue_label", "classification_status", "source_speech_text_hash"]],
        on="speech_id",
        how="left",
        validate="one_to_one",
    )
    if merged["issue_label"].isna().any():
        raise ValueError("one or more speeches are missing issue labels")
    return merged


def policy_speeches(frame: pd.DataFrame) -> pd.DataFrame:
    """Return speeches with a substantive policy label, excluding `NONE`."""
    return frame[frame["issue_label"].isin(POLICY_ISSUES)].copy()


def national_issue_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Counts and shares across all policy-labelled speeches in the supplied scope."""
    data = policy_speeches(frame)
    total = int(data["speech_id"].nunique())
    counts = data.groupby("issue_label")["speech_id"].nunique().reindex(POLICY_ISSUES, fill_value=0)
    result = counts.rename("issue_speech_count").reset_index()
    result["policy_speech_count"] = total
    result["issue_share"] = result["issue_speech_count"] / total if total else pd.NA
    return result


def grouped_issue_metrics(frame: pd.DataFrame, *, group_col: str) -> pd.DataFrame:
    """Return complete group × issue counts/shares, including zero issue counts."""
    data = policy_speeches(frame)
    groups = sorted(frame[group_col].dropna().astype(str).unique().tolist())
    if not groups:
        return pd.DataFrame(columns=[group_col, "issue_label", "issue_speech_count", "policy_speech_count", "issue_share"])

    universe = pd.MultiIndex.from_product([groups, POLICY_ISSUES], names=[group_col, "issue_label"]).to_frame(index=False)
    observed = (
        data[data[group_col].notna()]
        .groupby([group_col, "issue_label"])["speech_id"]
        .nunique()
        .rename("issue_speech_count")
        .reset_index()
    )
    totals = (
        data[data[group_col].notna()]
        .groupby(group_col)["speech_id"]
        .nunique()
        .rename("policy_speech_count")
        .reset_index()
    )
    result = universe.merge(observed, on=[group_col, "issue_label"], how="left")
    result = result.merge(totals, on=group_col, how="left")
    result["issue_speech_count"] = result["issue_speech_count"].fillna(0).astype(int)
    result["policy_speech_count"] = result["policy_speech_count"].fillna(0).astype(int)
    result["issue_share"] = result["issue_speech_count"].div(result["policy_speech_count"].replace(0, pd.NA)).astype("Float64")
    return result


def reliability_status(policy_speech_count: int) -> str:
    if policy_speech_count >= 20:
        return "reliable"
    if policy_speech_count >= 10:
        return "caution"
    return "insufficient_for_comparison"


def party_issue_comparisons(
    party_issue: pd.DataFrame,
    td_national_issue: pd.DataFrame,
    *,
    party_col: str = "party_uri",
    excluded_average_party_ids: set[str] | None = None,
    baseline_min_policy_speeches: int = 20,
) -> pd.DataFrame:
    """Add TD-national and average-party comparisons to party issue shares.

    The TD-national baseline is weighted naturally by all eligible TD policy
    speeches. The average-party baseline is an unweighted mean across parties
    with at least `baseline_min_policy_speeches`, excluding explicitly synthetic
    groupings such as the analytical Independent grouping.
    """
    excluded_average_party_ids = excluded_average_party_ids or set()
    required = {party_col, "issue_label", "issue_share", "policy_speech_count"}
    missing = sorted(required - set(party_issue.columns))
    if missing:
        raise ValueError(f"party issue metrics missing columns: {missing}")

    national = td_national_issue[["issue_label", "issue_share"]].rename(columns={"issue_share": "td_national_issue_share"})
    result = party_issue.merge(national, on="issue_label", how="left", validate="many_to_one")

    eligible = result[
        (result["policy_speech_count"] >= baseline_min_policy_speeches)
        & (~result[party_col].astype(str).isin(excluded_average_party_ids))
    ].copy()
    average = eligible.groupby("issue_label")["issue_share"].mean().rename("average_party_issue_share").reset_index()
    result = result.merge(average, on="issue_label", how="left", validate="many_to_one")

    result["share_vs_td_national_pp"] = (result["issue_share"] - result["td_national_issue_share"]) * 100.0
    result["share_vs_average_party_pp"] = (result["issue_share"] - result["average_party_issue_share"]) * 100.0
    result["emphasis_index_vs_td_national"] = 100.0 * result["issue_share"].div(result["td_national_issue_share"].replace(0, pd.NA))
    result["reliability_status"] = result["policy_speech_count"].map(lambda value: reliability_status(int(value)))
    result["comparison_public_safe"] = result["reliability_status"].eq("reliable")
    result["average_party_baseline_min_policy_speeches"] = baseline_min_policy_speeches
    return result
