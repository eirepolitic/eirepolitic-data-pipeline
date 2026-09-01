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
    overlap_examples: list[dict]

    def as_dict(self) -> dict:
        return asdict(self)


def _serialize_records(frame: pd.DataFrame, columns: list[str], *, limit: int = 10) -> list[dict]:
    existing = [col for col in columns if col in frame.columns]
    if not existing:
        return []
    rows = frame[existing].head(limit).copy()
    for col in rows.columns:
        if pd.api.types.is_datetime64_any_dtype(rows[col]):
            rows[col] = rows[col].dt.strftime("%Y-%m-%d")
    return rows.where(pd.notna(rows), None).to_dict(orient="records")


def temporal_overlap_examples(
    history: pd.DataFrame,
    *,
    entity_col: str,
    start_col: str,
    end_col: str,
    detail_columns: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Return rows involved in overlapping temporal intervals for diagnosis."""
    if history.empty:
        return []
    data = history.copy()
    data[start_col] = pd.to_datetime(data[start_col], errors="coerce").dt.normalize()
    data[end_col] = pd.to_datetime(data[end_col], errors="coerce").dt.normalize()
    output: list[dict] = []
    detail_columns = detail_columns or []

    for entity, group in data.dropna(subset=[start_col]).groupby(entity_col, dropna=False):
        ordered = group.sort_values(start_col)
        rows = list(ordered.to_dict(orient="records"))
        for idx, current in enumerate(rows):
            current_start = current[start_col]
            current_end = current[end_col]
            for later in rows[idx + 1 :]:
                later_start = later[start_col]
                later_end = later[end_col]
                overlaps = pd.isna(current_end) or later_start <= current_end
                if not overlaps:
                    break
                record = {entity_col: entity}
                for prefix, source in (("left", current), ("right", later)):
                    record[f"{prefix}_{start_col}"] = source[start_col].strftime("%Y-%m-%d") if pd.notna(source[start_col]) else None
                    record[f"{prefix}_{end_col}"] = source[end_col].strftime("%Y-%m-%d") if pd.notna(source[end_col]) else None
                    for col in detail_columns:
                        if col in source:
                            value = source[col]
                            if isinstance(value, pd.Timestamp):
                                value = value.strftime("%Y-%m-%d")
                            record[f"{prefix}_{col}"] = None if pd.isna(value) else value
                output.append(record)
                if len(output) >= limit:
                    return output
    return output


def history_coverage(
    history: pd.DataFrame,
    *,
    dataset: str,
    entity_col: str,
    start_col: str,
    end_col: str,
    detail_columns: list[str] | None = None,
) -> HistoryCoverage:
    """Summarise temporal coverage without claiming completeness beyond the data."""
    if history.empty:
        return HistoryCoverage(dataset, 0, 0, None, None, 0, [], [])

    data = history.copy()
    starts = pd.to_datetime(data[start_col], errors="coerce")
    ends = pd.to_datetime(data[end_col], errors="coerce")
    errors = validate_temporal_history(
        data,
        entity_col=entity_col,
        start_col=start_col,
        end_col=end_col,
    )
    overlaps = temporal_overlap_examples(
        data,
        entity_col=entity_col,
        start_col=start_col,
        end_col=end_col,
        detail_columns=detail_columns,
    )
    return HistoryCoverage(
        dataset=dataset,
        row_count=int(len(data)),
        entity_count=int(data[entity_col].nunique(dropna=True)),
        min_start=starts.min().date().isoformat() if starts.notna().any() else None,
        max_end=ends.max().date().isoformat() if ends.notna().any() else None,
        open_ended_rows=int(ends.isna().sum()),
        validation_errors=errors,
        overlap_examples=overlaps,
    )


def _unmatched_examples(joined: pd.DataFrame, history_value_col: str, *, limit: int = 10) -> list[dict]:
    columns = [
        "speech_id", "debate_id", "member_code", "event_date", "speaker_name",
        "speaker_ref", "speaker_match_method", "speaker_match_confidence",
    ]
    return _serialize_records(joined.loc[joined[history_value_col].isna()], columns, limit=limit)


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
