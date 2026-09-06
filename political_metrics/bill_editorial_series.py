from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd

CORE_STATUSES = {"current", "enacted"}

_STAGE_LABELS = {
    "first stage": ("first_stage", "First Stage", 20),
    "second stage": ("second_stage", "Second Stage", 30),
    "committee stage": ("committee_stage", "Committee Stage", 40),
    "report stage": ("report_stage", "Report Stage", 50),
    "fifth stage": ("fifth_stage", "Fifth Stage", 60),
    "cream list": ("returned_amendments", "Returned amendments", 70),
}


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].fillna("").astype(str).str.strip()
    return out


def _editorial_bucket(status: str, stage_name: str) -> tuple[str, str, int]:
    status_key = (status or "").casefold()
    if status_key == "enacted":
        return "enacted", "Enacted", 0
    if status_key == "defeated":
        return "defeated", "Defeated", 80
    if status_key == "lapsed":
        return "lapsed", "Lapsed", 90
    if status_key == "withdrawn":
        return "withdrawn", "Withdrawn", 91
    if status_key != "current":
        return "review_required", status or "Status needs review", 99
    return _STAGE_LABELS.get((stage_name or "").casefold(), ("review_required", stage_name or "Stage needs review", 99))


def build_editorial_bill_series(
    snapshot: pd.DataFrame,
    *,
    batch_size: int = 6,
    as_of_date: date,
    lookback_days: int = 180,
    previous_snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the Bills worth covering in one tracker edition.

    First edition: Current or Enacted Bills with source activity inside the
    lookback window. Later editions: Current or Enacted Bills whose deterministic
    state key is new or changed since the previous snapshot.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    df = _clean(snapshot)
    required = {"bill_id", "status", "current_stage_name", "current_state_key", "last_event_date", "bill_year", "bill_no"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"snapshot missing required columns: {missing}")

    df["change_type"] = "baseline_recent"
    mode = "baseline_recent"
    if previous_snapshot is not None and not previous_snapshot.empty:
        prev = _clean(previous_snapshot)
        if not {"bill_id", "current_state_key"}.issubset(prev.columns):
            raise ValueError("previous snapshot must contain bill_id and current_state_key")
        prev_state = prev.drop_duplicates("bill_id").set_index("bill_id")["current_state_key"].to_dict()
        def classify(row):
            old = prev_state.get(row["bill_id"])
            if old is None:
                return "new_bill"
            if old != row["current_state_key"]:
                return "state_changed"
            return "unchanged"
        df["change_type"] = df.apply(classify, axis=1)
        df = df[df["change_type"].isin(["new_bill", "state_changed"])].copy()
        mode = "snapshot_delta"
    else:
        cutoff = pd.Timestamp(as_of_date - timedelta(days=lookback_days))
        last_event = pd.to_datetime(df["last_event_date"], errors="coerce")
        df = df[last_event.ge(cutoff)].copy()

    df = df[df["status"].str.casefold().isin(CORE_STATUSES)].copy()
    if df.empty:
        df["editorial_scope_mode"] = mode
        return df

    bucket_values = df.apply(lambda r: _editorial_bucket(r["status"], r["current_stage_name"]), axis=1)
    df["editorial_bucket"] = [x[0] for x in bucket_values]
    df["editorial_bucket_label"] = [x[1] for x in bucket_values]
    df["editorial_bucket_order"] = [x[2] for x in bucket_values]
    df["editorial_scope_mode"] = mode
    df["editorial_as_of_date"] = as_of_date.isoformat()
    df["editorial_lookback_days"] = lookback_days

    df["_stage_date"] = pd.to_datetime(df.get("current_stage_date", ""), errors="coerce")
    df["_year"] = pd.to_numeric(df["bill_year"], errors="coerce")
    df["_no"] = pd.to_numeric(df["bill_no"], errors="coerce")
    df = df.sort_values(
        ["editorial_bucket_order", "_stage_date", "_year", "_no", "bill_id"],
        ascending=[True, False, False, False, True], kind="stable"
    ).reset_index(drop=True)

    df["editorial_batch_no"] = 0
    df["editorial_batch_count"] = 0
    df["editorial_position"] = 0
    for bucket, indexes in df.groupby("editorial_bucket", sort=False).groups.items():
        idx = list(indexes)
        total = len(idx)
        batch_count = int(math.ceil(total / batch_size))
        for sequence, row_idx in enumerate(idx, start=1):
            df.at[row_idx, "editorial_batch_no"] = int(math.ceil(sequence / batch_size))
            df.at[row_idx, "editorial_batch_count"] = batch_count
            df.at[row_idx, "editorial_position"] = ((sequence - 1) % batch_size) + 1
    df["editorial_batch_id"] = df.apply(
        lambda r: f"{r['editorial_bucket']}-{int(r['editorial_batch_no']):02d}", axis=1
    )
    return df.drop(columns=["_stage_date", "_year", "_no"]).reset_index(drop=True)


def audit_editorial_bill_series(frame: pd.DataFrame, *, batch_size: int = 6) -> dict:
    df = _clean(frame)
    if df.empty:
        return {
            "ready": True,
            "checks": {"empty_is_valid": True},
            "bill_count": 0,
            "batch_count": 0,
            "bucket_counts": {},
        }
    checks = {
        "core_status_only": bool(df["status"].str.casefold().isin(CORE_STATUSES).all()),
        "batch_size_respected": bool((pd.to_numeric(df["editorial_position"], errors="coerce") <= batch_size).all()),
        "one_row_per_bill": not df["bill_id"].duplicated().any(),
        "no_lapsed_or_withdrawn": not df["status"].str.casefold().isin(["lapsed", "withdrawn"]).any(),
        "cream_list_public_label": bool(
            df.loc[df["current_stage_name"].str.casefold().eq("cream list"), "editorial_bucket"]
            .eq("returned_amendments").all()
        ),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "bill_count": int(len(df)),
        "batch_count": int(df["editorial_batch_id"].nunique()),
        "bucket_counts": df["editorial_bucket"].value_counts().to_dict(),
        "change_type_counts": df["change_type"].value_counts().to_dict(),
    }
