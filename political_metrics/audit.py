from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .temporal_joins import attach_event_constituency, attach_event_membership, attach_event_party
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
    end_boundary: str = "exclusive",
) -> list[dict]:
    """Return rows involved in true temporal overlaps for diagnosis."""
    if end_boundary not in {"exclusive", "inclusive"}:
        raise ValueError(f"unsupported end boundary: {end_boundary}")
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
            current_end = current[end_col]
            for later in rows[idx + 1 :]:
                later_start = later[start_col]
                if pd.isna(current_end):
                    overlaps = True
                elif end_boundary == "exclusive":
                    overlaps = later_start < current_end
                else:
                    overlaps = later_start <= current_end
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
        end_boundary="exclusive",
    )
    overlaps = temporal_overlap_examples(
        data,
        entity_col=entity_col,
        start_col=start_col,
        end_col=end_col,
        detail_columns=detail_columns,
        end_boundary="exclusive",
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


def _speech_examples(frame: pd.DataFrame, *, limit: int = 10) -> list[dict]:
    columns = [
        "speech_id", "debate_id", "member_code", "event_date", "speaker_name",
        "speaker_ref", "speaker_match_method", "speaker_match_confidence",
    ]
    return _serialize_records(frame, columns, limit=limit)


def _unmatched_examples(joined: pd.DataFrame, history_value_col: str, *, limit: int = 10) -> list[dict]:
    if history_value_col not in joined.columns:
        return []
    return _speech_examples(joined.loc[joined[history_value_col].isna()], limit=limit)


def speech_temporal_attribution_audit(
    speeches: pd.DataFrame,
    member_parties: pd.DataFrame,
    member_constituencies: pd.DataFrame,
    member_memberships: pd.DataFrame | None = None,
    *,
    speech_date_col: str = "debate_date",
) -> dict:
    """Measure how completely TD speeches receive historical political context.

    Identified speakers who never appear in the Dáil membership history are kept
    as out-of-scope identified speakers rather than forced into TD statistics.
    Identified members who do exist in the membership table but lack a membership
    covering the speech date are treated as history gaps.
    """
    base = speeches.copy()
    base = base[base["member_code"].notna()].copy()
    base["event_date"] = base[speech_date_col]

    if member_memberships is not None:
        known_dail_codes = set(member_memberships["member_code"].dropna().astype(str))
        base_codes = base["member_code"].astype(str)
        out_of_scope = base.loc[~base_codes.isin(known_dail_codes)].copy()
        candidate_td = base.loc[base_codes.isin(known_dail_codes)].copy()
        membership_joined = attach_event_membership(candidate_td, member_memberships, event_date_col="event_date")
        membership_gap = membership_joined.loc[membership_joined["membership_id"].isna()].copy()
        td_speeches = membership_joined.loc[membership_joined["membership_id"].notna()].copy()
        if "chamber" in td_speeches.columns:
            td_speeches = td_speeches[td_speeches["chamber"].fillna("").str.lower().eq("dail")].copy()
    else:
        out_of_scope = base.iloc[0:0].copy()
        membership_gap = base.iloc[0:0].copy()
        td_speeches = base.copy()

    if td_speeches.empty:
        party_joined = td_speeches.assign(party_uri=pd.Series(dtype="object"))
        constituency_joined = td_speeches.assign(constituency_uri=pd.Series(dtype="object"))
    else:
        party_joined = attach_event_party(td_speeches, member_parties, event_date_col="event_date")
        constituency_joined = attach_event_constituency(td_speeches, member_constituencies, event_date_col="event_date")

    return {
        "identified_speech_rows": int(len(base)),
        "eligible_td_speech_rows": int(len(td_speeches)),
        "out_of_scope_identified_speech_rows": int(len(out_of_scope)),
        "membership_gap_speech_rows": int(len(membership_gap)),
        "out_of_scope_identified_speech_examples": _speech_examples(out_of_scope),
        "membership_gap_speech_examples": _speech_examples(membership_gap),
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
    """Reconcile national speech totals with identified and unidentified speech rows."""
    total = int(speeches[speech_id_col].dropna().nunique()) if speech_id_col in speeches.columns else 0
    attributable = speeches.loc[speeches["member_code"].notna(), speech_id_col].dropna().nunique() if "member_code" in speeches.columns else 0
    attributable = int(attributable)
    unattributed = total - attributable
    return {
        "national_distinct_speeches": total,
        "identified_member_speeches": attributable,
        "unidentified_speeches": unattributed,
        "member_attribution_coverage": (attributable / total) if total else 1.0,
        "reconciles": total == attributable + unattributed,
    }
