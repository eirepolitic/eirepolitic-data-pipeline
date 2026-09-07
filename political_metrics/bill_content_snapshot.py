from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pandas as pd

BATCH_SIZE_DEFAULT = 6

_STAGE_BUCKETS = {
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


def _latest_stage_rows(stages: pd.DataFrame) -> pd.DataFrame:
    st = _clean(stages)
    required = {"bill_id", "stage_name", "stage_date", "house_name", "order_in_bill"}
    missing = sorted(required - set(st.columns))
    if missing:
        raise ValueError(f"bill stages missing required columns: {missing}")
    if st.empty:
        return st
    st["_stage_date"] = pd.to_datetime(st["stage_date"], errors="coerce")
    st["_stage_order"] = pd.to_numeric(st["order_in_bill"], errors="coerce")
    st["_row_order"] = range(len(st))
    st = st.sort_values(["bill_id", "_stage_date", "_stage_order", "_row_order"], kind="stable")
    return st.groupby("bill_id", as_index=False).tail(1).copy()


def _sponsor_rollup(sponsors: pd.DataFrame) -> pd.DataFrame:
    sp = _clean(sponsors)
    required = {"bill_id", "sponsor_name", "sponsor_role_name", "sponsor_uri", "is_primary", "sponsor_order"}
    missing = sorted(required - set(sp.columns))
    if missing:
        raise ValueError(f"bill sponsors missing required columns: {missing}")
    if sp.empty:
        return pd.DataFrame(columns=[
            "bill_id", "sponsor_count", "primary_sponsor_name", "primary_sponsor_role_name",
            "primary_sponsor_uri", "sponsor_names_json", "sponsor_roles_json", "sponsor_attribution_status",
        ])

    sp["_order"] = pd.to_numeric(sp["sponsor_order"], errors="coerce")
    rows: list[dict] = []
    for bill_id, group in sp.groupby("bill_id", sort=True):
        group = group.sort_values(["_order", "sponsor_name"], kind="stable")
        primary = group[group["is_primary"].str.casefold().eq("true")]
        chosen = (primary if not primary.empty else group).iloc[0]
        names = sorted({v for v in group["sponsor_name"] if v})
        roles = sorted({v for v in group["sponsor_role_name"] if v})
        uri_rows = int(group["sponsor_uri"].ne("").sum())
        if uri_rows == len(group):
            attribution = "source_uri_present"
        elif uri_rows == 0 and roles:
            attribution = "role_or_office_only"
        elif uri_rows > 0:
            attribution = "mixed"
        else:
            attribution = "limited"
        rows.append({
            "bill_id": bill_id,
            "sponsor_count": int(len(group)),
            "primary_sponsor_name": chosen["sponsor_name"],
            "primary_sponsor_role_name": chosen["sponsor_role_name"],
            "primary_sponsor_uri": chosen["sponsor_uri"],
            "sponsor_names_json": json.dumps(names, ensure_ascii=False, separators=(",", ":")),
            "sponsor_roles_json": json.dumps(roles, ensure_ascii=False, separators=(",", ":")),
            "sponsor_attribution_status": attribution,
        })
    return pd.DataFrame(rows)


def _context_rollup(
    *, bridge: pd.DataFrame, speeches: pd.DataFrame, divisions: pd.DataFrame, member_votes: pd.DataFrame
) -> pd.DataFrame:
    br = _clean(bridge)
    sp = _clean(speeches)
    dv = _clean(divisions)
    mv = _clean(member_votes)
    required_bridge = {"bill_id", "debate_section_id"}
    missing = sorted(required_bridge - set(br.columns))
    if missing:
        raise ValueError(f"bill debate sections missing required columns: {missing}")
    if br.empty:
        return pd.DataFrame(columns=[
            "bill_id", "certified_section_count", "certified_speech_count", "certified_division_count",
            "latest_division_id", "latest_division_date", "latest_division_subject", "latest_division_outcome",
            "latest_division_ta", "latest_division_nil", "latest_division_abstain", "vote_breakdown_status",
        ])

    base = br[["bill_id", "debate_section_id"]].drop_duplicates()
    section_counts = base.groupby("bill_id").size().rename("certified_section_count")

    if {"debate_section_id", "speech_id"}.issubset(sp.columns):
        speech_counts = (
            sp[["speech_id", "debate_section_id"]]
            .merge(base, on="debate_section_id", how="inner", validate="many_to_one")
            .groupby("bill_id")["speech_id"].nunique()
            .rename("certified_speech_count")
        )
    else:
        speech_counts = pd.Series(dtype="int64", name="certified_speech_count")

    division_rows = pd.DataFrame()
    if {"division_id", "debate_section_id", "division_date"}.issubset(dv.columns):
        division_rows = dv.merge(base, on="debate_section_id", how="inner", validate="many_to_one")
    division_counts = (
        division_rows.groupby("bill_id")["division_id"].nunique().rename("certified_division_count")
        if not division_rows.empty else pd.Series(dtype="int64", name="certified_division_count")
    )

    latest = pd.DataFrame(columns=["bill_id"])
    if not division_rows.empty:
        division_rows = division_rows.copy()
        division_rows["_division_date"] = pd.to_datetime(division_rows["division_date"], errors="coerce")
        division_rows = division_rows.sort_values(["bill_id", "_division_date", "division_id"], kind="stable")
        latest = division_rows.groupby("bill_id", as_index=False).tail(1).copy()
        rename = {
            "division_id": "latest_division_id",
            "division_date": "latest_division_date",
            "subject": "latest_division_subject",
            "outcome": "latest_division_outcome",
        }
        keep = ["bill_id"] + [c for c in rename if c in latest.columns]
        latest = latest[keep].rename(columns=rename)

        if {"division_id", "vote_label"}.issubset(mv.columns):
            votes = mv.copy()
            votes["_kind"] = votes["vote_label"].str.casefold().map({
                "yes": "ta", "ta": "ta", "tá": "ta",
                "no": "nil", "nil": "nil", "níl": "nil",
                "abstain": "abstain", "staon": "abstain",
            }).fillna("other")
            counts = votes.groupby(["division_id", "_kind"]).size().unstack(fill_value=0).reset_index()
            for col in ["ta", "nil", "abstain"]:
                if col not in counts.columns:
                    counts[col] = 0
            counts = counts.rename(columns={
                "division_id": "latest_division_id",
                "ta": "latest_division_ta",
                "nil": "latest_division_nil",
                "abstain": "latest_division_abstain",
            })
            latest = latest.merge(
                counts[["latest_division_id", "latest_division_ta", "latest_division_nil", "latest_division_abstain"]],
                on="latest_division_id", how="left", validate="one_to_one"
            )

    out = pd.concat([section_counts, speech_counts, division_counts], axis=1).fillna(0).reset_index()
    out = out.merge(latest, on="bill_id", how="left", validate="one_to_one")
    for col in ["certified_section_count", "certified_speech_count", "certified_division_count", "latest_division_ta", "latest_division_nil", "latest_division_abstain"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["vote_breakdown_status"] = out["certified_division_count"].map(lambda n: "recorded_division_available" if n > 0 else "no_recorded_division")
    return out


def _bucket_for(status: str, stage_name: str) -> tuple[str, str, int]:
    status_key = (status or "").casefold()
    terminal = {
        "enacted": ("enacted", "Enacted", 0),
        "defeated": ("defeated", "Defeated", 80),
        "lapsed": ("lapsed", "Lapsed", 90),
        "withdrawn": ("withdrawn", "Withdrawn", 91),
    }
    if status_key in terminal:
        return terminal[status_key]
    stage_key = (stage_name or "").casefold()
    if stage_key in _STAGE_BUCKETS:
        return _STAGE_BUCKETS[stage_key]
    return "review_required", stage_name or status or "Stage needs review", 99


def build_bill_content_snapshot(
    *,
    bills: pd.DataFrame,
    stages: pd.DataFrame,
    sponsors: pd.DataFrame,
    bill_debate_sections: pd.DataFrame,
    speeches: pd.DataFrame,
    divisions: pd.DataFrame,
    member_votes: pd.DataFrame,
    batch_size: int = BATCH_SIZE_DEFAULT,
    generated_at_utc: str | None = None,
) -> pd.DataFrame:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    bill = _clean(bills)
    required_bills = {
        "bill_id", "bill_no", "bill_year", "title", "short_title", "origin_house_name",
        "bill_type", "status", "introduced_date", "last_event_date", "snapshot_date",
    }
    missing = sorted(required_bills - set(bill.columns))
    if missing:
        raise ValueError(f"bills missing required columns: {missing}")
    if bill.empty:
        return pd.DataFrame()

    latest = _latest_stage_rows(stages)
    latest_cols = ["bill_id", "stage_name", "stage_date", "house_name", "stage_outcome", "order_in_bill"]
    latest = latest[[c for c in latest_cols if c in latest.columns]].rename(columns={
        "stage_name": "current_stage_name",
        "stage_date": "current_stage_date",
        "house_name": "current_stage_house_name",
        "stage_outcome": "current_stage_outcome",
        "order_in_bill": "current_stage_order",
    })
    out = bill.merge(latest, on="bill_id", how="left", validate="one_to_one")
    out = out.merge(_sponsor_rollup(sponsors), on="bill_id", how="left", validate="one_to_one")
    out = out.merge(
        _context_rollup(bridge=bill_debate_sections, speeches=speeches, divisions=divisions, member_votes=member_votes),
        on="bill_id", how="left", validate="one_to_one"
    )

    for col in ["sponsor_count", "certified_section_count", "certified_speech_count", "certified_division_count", "latest_division_ta", "latest_division_nil", "latest_division_abstain"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in [
        "primary_sponsor_name", "primary_sponsor_role_name", "primary_sponsor_uri", "sponsor_names_json",
        "sponsor_roles_json", "sponsor_attribution_status", "latest_division_id", "latest_division_date",
        "latest_division_subject", "latest_division_outcome", "vote_breakdown_status",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    buckets = out.apply(lambda r: _bucket_for(r.get("status", ""), r.get("current_stage_name", "")), axis=1)
    out["series_bucket"] = [b[0] for b in buckets]
    out["series_bucket_label"] = [b[1] for b in buckets]
    out["series_bucket_order"] = [b[2] for b in buckets]
    out["series_group_label"] = out["series_bucket_label"]
    out["house_badge"] = out["current_stage_house_name"].where(out["current_stage_house_name"].ne(""), out["origin_house_name"])

    out["_stage_date_sort"] = pd.to_datetime(out["current_stage_date"], errors="coerce")
    out["_bill_year_sort"] = pd.to_numeric(out["bill_year"], errors="coerce")
    out["_bill_no_sort"] = pd.to_numeric(out["bill_no"], errors="coerce")
    out = out.sort_values(
        ["series_bucket_order", "_stage_date_sort", "_bill_year_sort", "_bill_no_sort", "bill_id"],
        ascending=[True, False, False, False, True], kind="stable"
    ).reset_index(drop=True)

    out["series_batch_no"] = 0
    out["series_batch_count"] = 0
    out["position_in_batch"] = 0
    for bucket, idx in out.groupby("series_bucket", sort=False).groups.items():
        positions = list(idx)
        total = len(positions)
        batch_count = int(math.ceil(total / batch_size))
        for sequence, row_idx in enumerate(positions, start=1):
            out.at[row_idx, "series_batch_no"] = int(math.ceil(sequence / batch_size))
            out.at[row_idx, "series_batch_count"] = batch_count
            out.at[row_idx, "position_in_batch"] = ((sequence - 1) % batch_size) + 1

    out["series_batch_id"] = out.apply(
        lambda r: f"{r['series_bucket']}-{int(r['series_batch_no']):02d}", axis=1
    )
    out["current_state_key"] = out.apply(
        lambda r: "|".join([
            str(r.get("status", "")), str(r.get("current_stage_name", "")),
            str(r.get("current_stage_house_name", "")), str(r.get("current_stage_date", "")),
        ]), axis=1
    )
    out["bill_summary_status"] = "editorial_summary_required"
    out["debate_summary_status"] = out["certified_speech_count"].map(
        lambda n: "certified_debate_available" if n > 0 else "no_certified_debate"
    )
    out["support_opposition_status"] = out["certified_division_count"].map(
        lambda n: "recorded_vote_evidence_available" if n > 0 else "do_not_infer_from_speeches"
    )
    out["generated_at_utc"] = generated_at_utc or datetime.now(timezone.utc).isoformat()
    out["batch_size"] = batch_size

    drop = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop).reset_index(drop=True)


def audit_bill_content_snapshot(frame: pd.DataFrame, *, batch_size: int = BATCH_SIZE_DEFAULT) -> dict:
    df = _clean(frame)
    if df.empty:
        return {"ready": False, "checks": {"nonempty": False}}
    checks = {
        "nonempty": len(df) > 0,
        "one_row_per_bill": not df["bill_id"].duplicated().any(),
        "batch_size_respected": bool((pd.to_numeric(df["position_in_batch"], errors="coerce") <= batch_size).all()),
        "state_key_populated": bool(df["current_state_key"].ne("").all()),
        "unsafe_support_inference_absent": bool(
            df.loc[pd.to_numeric(df["certified_division_count"], errors="coerce").fillna(0).eq(0), "support_opposition_status"]
            .eq("do_not_infer_from_speeches").all()
        ),
        "cream_list_relabelled": bool(
            df.loc[df["current_stage_name"].str.casefold().eq("cream list"), "series_bucket"]
            .eq("returned_amendments").all()
        ),
        "terminal_statuses_not_stage_bucketed": bool(
            df.loc[df["status"].str.casefold().isin(["enacted", "defeated", "lapsed", "withdrawn"]), "series_bucket"]
            .isin(["enacted", "defeated", "lapsed", "withdrawn"]).all()
        ),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "bill_count": int(len(df)),
        "status_counts": df["status"].value_counts().to_dict(),
        "series_bucket_counts": df["series_bucket"].value_counts().to_dict(),
        "series_batch_count": int(df["series_batch_id"].nunique()),
        "bills_with_certified_debate": int((pd.to_numeric(df["certified_speech_count"], errors="coerce").fillna(0) > 0).sum()),
        "bills_with_recorded_vote_evidence": int((pd.to_numeric(df["certified_division_count"], errors="coerce").fillna(0) > 0).sum()),
    }
