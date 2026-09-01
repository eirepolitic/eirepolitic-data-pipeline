from __future__ import annotations

import pandas as pd


def _normalise_days(values: pd.Series | list) -> pd.Series:
    return (
        pd.Series(pd.to_datetime(list(values), errors="coerce"))
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )


def member_debate_day_exposure(
    memberships: pd.DataFrame,
    debate_days: pd.Series | list,
    *,
    member_col: str = "member_code",
    start_col: str = "membership_start",
    end_col: str = "membership_end",
) -> pd.DataFrame:
    """Return eligible debate-day exposure for each member.

    Oireachtas ranges are treated as start-inclusive/end-exclusive. A member is
    eligible when ``start <= debate_day < end`` or the end is open.
    """
    days = _normalise_days(debate_days)
    if memberships.empty or days.empty:
        return pd.DataFrame(columns=[member_col, "eligible_debate_days"])

    history = memberships[[member_col, start_col, end_col]].copy()
    history[start_col] = pd.to_datetime(history[start_col], errors="coerce").dt.normalize()
    history[end_col] = pd.to_datetime(history[end_col], errors="coerce").dt.normalize()

    records: list[dict] = []
    for member, group in history.groupby(member_col, dropna=False):
        eligible_dates: set[pd.Timestamp] = set()
        for row in group.itertuples(index=False):
            start = getattr(row, start_col)
            end = getattr(row, end_col)
            if pd.isna(start):
                continue
            mask = days >= start
            if pd.notna(end):
                mask &= days < end
            eligible_dates.update(days[mask].tolist())
        records.append({member_col: member, "eligible_debate_days": len(eligible_dates)})

    return pd.DataFrame.from_records(records)


def group_member_debate_day_exposure(
    history: pd.DataFrame,
    debate_days: pd.Series | list,
    *,
    group_col: str,
    member_col: str = "member_code",
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    """Calculate period-correct member-day exposure for a party or constituency.

    Ranges are start-inclusive/end-exclusive. The output includes both raw
    member-debate-days and full-period active-member equivalents.
    """
    days = _normalise_days(debate_days)
    columns = [group_col, "member_debate_days", "active_member_equivalent", "active_member_count"]
    if history.empty or days.empty:
        return pd.DataFrame(columns=columns)

    required = {member_col, group_col, start_col, end_col}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"history missing required columns: {missing}")

    data = history[[member_col, group_col, start_col, end_col]].copy()
    data[start_col] = pd.to_datetime(data[start_col], errors="coerce").dt.normalize()
    data[end_col] = pd.to_datetime(data[end_col], errors="coerce").dt.normalize()
    data = data[data[group_col].notna() & data[member_col].notna() & data[start_col].notna()].copy()

    member_days: dict[tuple[object, object], set[pd.Timestamp]] = {}
    for row in data.itertuples(index=False):
        member = getattr(row, member_col)
        group = getattr(row, group_col)
        start = getattr(row, start_col)
        end = getattr(row, end_col)
        mask = days >= start
        if pd.notna(end):
            mask &= days < end
        member_days.setdefault((group, member), set()).update(days[mask].tolist())

    records: list[dict] = []
    total_days = len(days)
    groups: dict[object, list[tuple[object, set[pd.Timestamp]]]] = {}
    for (group, member), eligible_dates in member_days.items():
        if eligible_dates:
            groups.setdefault(group, []).append((member, eligible_dates))

    for group, members in groups.items():
        exposure = sum(len(eligible_dates) for _, eligible_dates in members)
        records.append({
            group_col: group,
            "member_debate_days": int(exposure),
            "active_member_equivalent": float(exposure) / float(total_days),
            "active_member_count": len(members),
        })

    return pd.DataFrame.from_records(records, columns=columns)


def party_debate_day_exposure(member_parties: pd.DataFrame, debate_days: pd.Series | list) -> pd.DataFrame:
    return group_member_debate_day_exposure(
        member_parties,
        debate_days,
        group_col="party_uri",
        start_col="party_start",
        end_col="party_end",
    )


def constituency_debate_day_exposure(member_constituencies: pd.DataFrame, debate_days: pd.Series | list) -> pd.DataFrame:
    return group_member_debate_day_exposure(
        member_constituencies,
        debate_days,
        group_col="constituency_uri",
        start_col="represent_start",
        end_col="represent_end",
    )


def active_member_equivalent(member_exposure: pd.DataFrame, total_debate_days: int) -> float:
    """Convert summed member debate-day exposure to full-period member equivalents."""
    if total_debate_days <= 0:
        return 0.0
    if member_exposure.empty:
        return 0.0
    return float(member_exposure["eligible_debate_days"].sum()) / float(total_debate_days)
