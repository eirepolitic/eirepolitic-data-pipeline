from __future__ import annotations

import pandas as pd


def member_debate_day_exposure(
    memberships: pd.DataFrame,
    debate_days: pd.Series | list,
    *,
    member_col: str = "member_code",
    start_col: str = "membership_start",
    end_col: str = "membership_end",
) -> pd.DataFrame:
    """Return eligible debate-day exposure for each member.

    A member is eligible on a debate day when the day falls within at least one
    active membership interval. Duplicate/overlapping membership rows do not
    double-count the same member-day.
    """
    days = pd.Series(pd.to_datetime(list(debate_days), errors="coerce")).dropna().dt.normalize().drop_duplicates().sort_values()
    if memberships.empty or days.empty:
        return pd.DataFrame(columns=[member_col, "eligible_debate_days"])

    history = memberships[[member_col, start_col, end_col]].copy()
    history[start_col] = pd.to_datetime(history[start_col], errors="coerce").dt.normalize()
    history[end_col] = pd.to_datetime(history[end_col], errors="coerce").dt.normalize()

    records: list[dict] = []
    for member, group in history.groupby(member_col, dropna=False):
        eligible = pd.Series(False, index=days.index)
        for row in group.itertuples(index=False):
            start = getattr(row, start_col)
            end = getattr(row, end_col)
            if pd.isna(start):
                continue
            mask = days >= start
            if pd.notna(end):
                mask &= days <= end
            eligible |= mask
        records.append({member_col: member, "eligible_debate_days": int(eligible.sum())})

    return pd.DataFrame.from_records(records)


def active_member_equivalent(member_exposure: pd.DataFrame, total_debate_days: int) -> float:
    """Convert summed member debate-day exposure to full-period member equivalents."""
    if total_debate_days <= 0:
        return 0.0
    if member_exposure.empty:
        return 0.0
    return float(member_exposure["eligible_debate_days"].sum()) / float(total_debate_days)
