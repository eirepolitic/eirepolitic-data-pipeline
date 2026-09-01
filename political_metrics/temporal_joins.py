from __future__ import annotations

from typing import Literal

import pandas as pd

EndBoundary = Literal["exclusive", "inclusive"]


def temporal_join(
    events: pd.DataFrame,
    history: pd.DataFrame,
    *,
    event_date_col: str,
    entity_col: str,
    history_start_col: str,
    history_end_col: str,
    history_columns: list[str] | None = None,
    allow_unmatched: bool = True,
    end_boundary: EndBoundary = "exclusive",
) -> pd.DataFrame:
    """Attach the single history row valid on each event date.

    Oireachtas history ranges default to start-inclusive/end-exclusive semantics:
    ``start <= event_date < end``. This matches observed party transitions where
    an old affiliation ends on the same date the next affiliation starts. Open-
    ended rows continue indefinitely. Ambiguous overlaps raise rather than being
    resolved silently.
    """
    if end_boundary not in {"exclusive", "inclusive"}:
        raise ValueError(f"unsupported end boundary: {end_boundary}")
    if events.empty:
        return events.copy()

    left = events.copy()
    right = history.copy()
    left["__event_row_id"] = range(len(left))
    left[event_date_col] = pd.to_datetime(left[event_date_col], errors="coerce").dt.normalize()
    right[history_start_col] = pd.to_datetime(right[history_start_col], errors="coerce").dt.normalize()
    right[history_end_col] = pd.to_datetime(right[history_end_col], errors="coerce").dt.normalize()

    keep = [entity_col, history_start_col, history_end_col]
    for col in history_columns or []:
        if col not in keep:
            keep.append(col)

    missing_history_columns = [col for col in keep if col not in right.columns]
    if missing_history_columns:
        raise ValueError(f"history missing required columns: {missing_history_columns}")

    merged = left.merge(right[keep], how="left", on=entity_col, suffixes=("", "__history"))
    event_date = merged[event_date_col]
    start = merged[history_start_col]
    end = merged[history_end_col]
    if end_boundary == "exclusive":
        end_valid = end.isna() | (event_date < end)
    else:
        end_valid = end.isna() | (event_date <= end)
    valid = start.notna() & (start <= event_date) & end_valid
    matched = merged.loc[valid].copy()

    ambiguous = matched.groupby("__event_row_id", dropna=False).size()
    ambiguous = ambiguous[ambiguous > 1]
    if not ambiguous.empty:
        examples = ambiguous.index[:10].tolist()
        raise ValueError(f"ambiguous temporal history matches for event rows: {examples}")

    if allow_unmatched:
        missing_ids = left.loc[~left["__event_row_id"].isin(matched["__event_row_id"]), "__event_row_id"]
        if not missing_ids.empty:
            unmatched = left[left["__event_row_id"].isin(missing_ids)].copy()
            for col in keep:
                if col != entity_col and col not in unmatched.columns:
                    unmatched[col] = pd.NA
            matched = pd.concat([matched, unmatched], ignore_index=True, sort=False)
    elif matched["__event_row_id"].nunique() != len(left):
        missing = len(left) - matched["__event_row_id"].nunique()
        raise ValueError(f"{missing} event rows have no valid temporal history match")

    result = matched.sort_values("__event_row_id").drop(columns="__event_row_id")
    return result.reset_index(drop=True)


def attach_event_membership(
    events: pd.DataFrame,
    member_memberships: pd.DataFrame,
    *,
    event_date_col: str = "event_date",
    member_col: str = "member_code",
) -> pd.DataFrame:
    return temporal_join(
        events,
        member_memberships,
        event_date_col=event_date_col,
        entity_col=member_col,
        history_start_col="membership_start",
        history_end_col="membership_end",
        history_columns=["membership_id", "house_uri", "house_no", "chamber"],
        end_boundary="exclusive",
    )


def attach_event_party(
    events: pd.DataFrame,
    member_parties: pd.DataFrame,
    *,
    event_date_col: str = "event_date",
    member_col: str = "member_code",
) -> pd.DataFrame:
    return temporal_join(
        events,
        member_parties,
        event_date_col=event_date_col,
        entity_col=member_col,
        history_start_col="party_start",
        history_end_col="party_end",
        history_columns=["party_uri", "party_name"],
        end_boundary="exclusive",
    )


def attach_event_constituency(
    events: pd.DataFrame,
    member_constituencies: pd.DataFrame,
    *,
    event_date_col: str = "event_date",
    member_col: str = "member_code",
) -> pd.DataFrame:
    return temporal_join(
        events,
        member_constituencies,
        event_date_col=event_date_col,
        entity_col=member_col,
        history_start_col="represent_start",
        history_end_col="represent_end",
        history_columns=["constituency_uri", "constituency_name"],
        end_boundary="exclusive",
    )
