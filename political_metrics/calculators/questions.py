from __future__ import annotations

import pandas as pd

from political_metrics.temporal_joins import attach_event_constituency, attach_event_membership, attach_event_party


def prepare_eligible_td_questions(
    questions: pd.DataFrame,
    memberships: pd.DataFrame,
    member_parties: pd.DataFrame,
    member_constituencies: pd.DataFrame,
) -> pd.DataFrame:
    """Attach active Dáil membership and historical context to question facts."""
    required = {"question_id", "question_date", "asked_by_member_code"}
    missing = sorted(required - set(questions.columns))
    if missing:
        raise ValueError(f"questions missing required columns: {missing}")

    data = questions.copy()
    data = data[data["asked_by_member_code"].notna()].copy()
    data = data.rename(columns={"asked_by_member_code": "member_code"})
    data["event_date"] = data["question_date"]

    known = set(memberships["member_code"].dropna().astype(str))
    data = data[data["member_code"].astype(str).isin(known)].copy()
    if data.empty:
        return data.assign(party_uri=pd.Series(dtype="object"), constituency_uri=pd.Series(dtype="object"))

    data = attach_event_membership(data, memberships, event_date_col="event_date")
    data = data[data["membership_id"].notna()].copy()
    if "chamber" in data.columns:
        data = data[data["chamber"].fillna("").str.lower().eq("dail")].copy()
    if data.empty:
        return data.assign(party_uri=pd.Series(dtype="object"), constituency_uri=pd.Series(dtype="object"))

    data = attach_event_party(data, member_parties, event_date_col="event_date")
    data = attach_event_constituency(data, member_constituencies, event_date_col="event_date")
    return data


def member_question_metrics(questions: pd.DataFrame) -> pd.DataFrame:
    """Question volume, type breadth, and recipient breadth for eligible TDs."""
    if questions.empty:
        return pd.DataFrame(columns=[
            "member_code", "question_count", "question_day_count",
            "question_type_count", "recipient_count",
        ])
    return (
        questions.groupby("member_code")
        .agg(
            question_count=("question_id", "nunique"),
            question_day_count=("question_date", "nunique"),
            question_type_count=("question_type", lambda s: s.dropna().astype(str).replace("", pd.NA).dropna().nunique()),
            recipient_count=("to_minister_or_department", lambda s: s.dropna().astype(str).replace("", pd.NA).dropna().nunique()),
        )
        .reset_index()
    )


def grouped_question_metrics(questions: pd.DataFrame, *, group_col: str) -> pd.DataFrame:
    """Question volume and participation for a historical party/constituency group."""
    if questions.empty:
        return pd.DataFrame(columns=[
            group_col, "question_count", "asking_member_count", "question_day_count",
            "question_type_count", "recipient_count",
        ])
    data = questions[questions[group_col].notna()].copy()
    return (
        data.groupby(group_col)
        .agg(
            question_count=("question_id", "nunique"),
            asking_member_count=("member_code", "nunique"),
            question_day_count=("question_date", "nunique"),
            question_type_count=("question_type", lambda s: s.dropna().astype(str).replace("", pd.NA).dropna().nunique()),
            recipient_count=("to_minister_or_department", lambda s: s.dropna().astype(str).replace("", pd.NA).dropna().nunique()),
        )
        .reset_index()
    )


def national_question_metrics(questions: pd.DataFrame) -> dict[str, int]:
    return {
        "question_count": int(questions["question_id"].nunique()) if not questions.empty else 0,
        "asking_member_count": int(questions["member_code"].nunique()) if not questions.empty and "member_code" in questions.columns else 0,
        "question_day_count": int(questions["question_date"].nunique()) if not questions.empty else 0,
        "question_type_count": int(questions["question_type"].replace("", pd.NA).dropna().nunique()) if not questions.empty else 0,
        "recipient_count": int(questions["to_minister_or_department"].replace("", pd.NA).dropna().nunique()) if not questions.empty else 0,
    }


def question_type_distribution(questions: pd.DataFrame, *, group_col: str | None = None) -> pd.DataFrame:
    """Return question-type counts/shares nationally or by a supplied group."""
    data = questions.copy()
    data["question_type"] = data["question_type"].fillna("").astype(str).str.strip()
    data = data[data["question_type"].ne("")].copy()
    keys = ["question_type"] if group_col is None else [group_col, "question_type"]
    counts = data.groupby(keys)["question_id"].nunique().rename("question_count").reset_index()
    if group_col is None:
        total = int(data["question_id"].nunique())
        counts["question_type_share"] = counts["question_count"] / total if total else pd.NA
    else:
        totals = data.groupby(group_col)["question_id"].nunique().rename("total_question_count").reset_index()
        counts = counts.merge(totals, on=group_col, how="left")
        counts["question_type_share"] = counts["question_count"].div(counts["total_question_count"].replace(0, pd.NA)).astype("Float64")
    return counts


def recipient_distribution(questions: pd.DataFrame, *, group_col: str | None = None) -> pd.DataFrame:
    """Return counts/shares by the recorded minister or department asked."""
    data = questions.copy()
    data["to_minister_or_department"] = data["to_minister_or_department"].fillna("").astype(str).str.strip()
    data = data[data["to_minister_or_department"].ne("")].copy()
    keys = ["to_minister_or_department"] if group_col is None else [group_col, "to_minister_or_department"]
    counts = data.groupby(keys)["question_id"].nunique().rename("question_count").reset_index()
    if group_col is None:
        total = int(data["question_id"].nunique())
        counts["question_share"] = counts["question_count"] / total if total else pd.NA
    else:
        totals = data.groupby(group_col)["question_id"].nunique().rename("total_question_count").reset_index()
        counts = counts.merge(totals, on=group_col, how="left")
        counts["question_share"] = counts["question_count"].div(counts["total_question_count"].replace(0, pd.NA)).astype("Float64")
    return counts
