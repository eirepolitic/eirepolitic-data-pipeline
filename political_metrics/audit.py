from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .temporal_joins import attach_event_constituency, attach_event_party
from .validators import temporal_join_coverage, validate_temporal_history


@dataclass(frozen=True)
class HistoryCoverage:
    dataset: str
    row_count: int
    entity_count: int
    min_start: str | None
    max_end: str | None
    open_ended_rows: int
    validation_errors: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def history_coverage(
    history: pd.DataFrame,
    *,
    dataset: str,
    entity_col: str,
    start_col: str,
    end_col: str,
) -> HistoryCoverage:
    """Summarise temporal coverage without claiming completeness beyond the data."""
    if history.empty:
        return HistoryCoverage(dataset, 0, 0, None, None, 0, [])

    data = history.copy()
    starts = pd.to_datetime(data[start_col], errors="coerce")
    ends = pd.to_datetime(data[end_col], errors="coerce")
    errors = validate_temporal_history(
        data,
        entity_col=entity_col,
        start_col=start_col,
        end_col=end_col,
    )
    return HistoryCoverage(
        dataset=dataset,
        row_count=int(len(data)),
        entity_count=int(data[entity_col].nunique(dropna=True)),
        min_start=starts.min().date().isoformat() if starts.notna().any() else None,
        max_end=ends.max().date().isoformat() if ends.notna().any() else None,
        open_ended_rows=int(ends.isna().sum()),
        validation_errors=errors,
    )


def _unmatched_examples(joined: pd.DataFrame, history_value_col: str, *, limit: int = 10) -> list[dict]:
    columns = [col for col in ["speech_id", "member_code", "event_date", "speaker_name"] if col in joined.columns]
    if not columns:
        return []
    rows = joined.loc[joined[history_value_col].isna(), columns].head(limit).copy()
    for col in rows.columns:
        if pd.api.types.is_datetime64_any_dtype(rows[col]):
            rows[col] = rows[col].dt.strftime("%Y-%m-%d")
    return rows.where(pd.notna(rows), None).to_dict(orient="records")


def speech_temporal_attribution_audit(
    speeches: pd.DataFrame,
    member_parties: pd.DataFrame,
    member_constituencies: pd.DataFrame,
    *,
    speech_date_col: str = "debate_date",
) -> dict:
    """Measure how completely speeches can be assigned to historical political context."""
    base = speeches.copy()
    base = base[base["member_code"].notna()].copy()
    base["event_date"] = base[speech_date_col]

    party_joined = attach_event_party(base, member_parties, event_date_col="event_date")
    constituency_joined = attach_event_constituency(base, member_constituencies, event_date_col="event_date")

    total_rows = int(len(base))
    return {
        "attributable_speech_rows": total_rows,
        "party_attribution_coverage": temporal_join_coverage(party_joined, "party_uri"),
        "constituency_attribution_coverage": temporal_join_coverage(constituency_joined, "constituency_uri"),
        "party_unmatched_rows": int(party_joined["party_uri"].isna().sum()),
        "constituency_unmatched_rows": int(constituency_joined["constituency_uri"].isna().sum()),
        "party_unmatched_examples": _unmatched_examples(party_joined, "party_uri"),
        "constituency_unmatched_examples": _unmatched_examples(constituency_joined, "constituency_uri"),
    }


def speech_count_reconciliation(
    speeches: pd.DataFrame,
    *,
    speech_id_col: str = "speech_id",
) -> dict:
    """Reconcile national speech totals with attributable and unattributed member speech rows."""
    total = int(speeches[speech_id_col].dropna().nunique()) if speech_id_col in speeches.columns else 0
    attributable = speeches.loc[speeches["member_code"].notna(), speech_id_col].dropna().nunique() if "member_code" in speeches.columns else 0
    attributable = int(attributable)
    unattributed = total - attributable
    return {
        "national_distinct_speeches": total,
        "attributable_member_speeches": attributable,
        "unattributed_speeches": unattributed,
        "member_attribution_coverage": (attributable / total) if total else 1.0,
        "reconciles": total == attributable + unattributed,
    }
