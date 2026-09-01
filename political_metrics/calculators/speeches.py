from __future__ import annotations

import pandas as pd


def _prepare_speeches(speeches: pd.DataFrame, *, date_col: str, speech_id_col: str) -> pd.DataFrame:
    required = {date_col, speech_id_col, "member_code"}
    missing = sorted(required - set(speeches.columns))
    if missing:
        raise ValueError(f"speeches missing required columns: {missing}")
    data = speeches.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    data = data[data[speech_id_col].notna() & data[date_col].notna()].copy()
    return data


def member_speech_metrics(
    speeches: pd.DataFrame,
    member_exposure: pd.DataFrame | None = None,
    *,
    date_col: str = "debate_date",
    speech_id_col: str = "speech_id",
) -> pd.DataFrame:
    """Calculate the first public member-level speech measures."""
    data = _prepare_speeches(speeches, date_col=date_col, speech_id_col=speech_id_col)
    attributable = data[data["member_code"].notna()].copy()
    if attributable.empty:
        return pd.DataFrame(columns=[
            "member_code", "speech_count", "speaking_day_count",
            "share_of_dail_speeches", "eligible_debate_days",
            "speeches_per_eligible_debate_day",
        ])

    grouped = attributable.groupby("member_code", dropna=False).agg(
        speech_count=(speech_id_col, "nunique"),
        speaking_day_count=(date_col, "nunique"),
    ).reset_index()

    total = int(attributable[speech_id_col].nunique())
    grouped["share_of_dail_speeches"] = grouped["speech_count"] / total if total else 0.0

    if member_exposure is not None:
        grouped = grouped.merge(member_exposure[["member_code", "eligible_debate_days"]], on="member_code", how="left")
        grouped["eligible_debate_days"] = grouped["eligible_debate_days"].fillna(0).astype(int)
        grouped["speeches_per_eligible_debate_day"] = grouped["speech_count"].div(
            grouped["eligible_debate_days"].replace(0, pd.NA)
        ).astype("Float64")
    else:
        grouped["eligible_debate_days"] = pd.NA
        grouped["speeches_per_eligible_debate_day"] = pd.NA

    return grouped.sort_values(["speech_count", "member_code"], ascending=[False, True]).reset_index(drop=True)


def grouped_speech_metrics(
    speeches: pd.DataFrame,
    *,
    group_col: str,
    date_col: str = "debate_date",
    speech_id_col: str = "speech_id",
) -> pd.DataFrame:
    """Calculate reusable party/constituency speech counts after temporal attribution."""
    data = _prepare_speeches(speeches, date_col=date_col, speech_id_col=speech_id_col)
    data = data[data[group_col].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=[group_col, "speech_count", "speaking_member_count", "share_of_dail_speeches"])

    grouped = data.groupby(group_col, dropna=False).agg(
        speech_count=(speech_id_col, "nunique"),
        speaking_member_count=("member_code", "nunique"),
    ).reset_index()
    total = int(data[speech_id_col].nunique())
    grouped["share_of_dail_speeches"] = grouped["speech_count"] / total if total else 0.0
    return grouped.sort_values(["speech_count", group_col], ascending=[False, True]).reset_index(drop=True)


def national_speech_metrics(
    speeches: pd.DataFrame,
    *,
    date_col: str = "debate_date",
    speech_id_col: str = "speech_id",
) -> dict[str, int | float]:
    data = _prepare_speeches(speeches, date_col=date_col, speech_id_col=speech_id_col)
    debate_days = int(data[date_col].nunique())
    speech_count = int(data[speech_id_col].nunique())
    return {
        "speech_count": speech_count,
        "unique_speaker_count": int(data["member_code"].nunique()),
        "debate_day_count": debate_days,
        "speeches_per_debate_day": (speech_count / debate_days) if debate_days else 0.0,
    }
