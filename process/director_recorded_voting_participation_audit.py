#!/usr/bin/env python3
"""Read-only Director audit for recorded Dáil voting participation.

This script resolves one immutable production batch, validates the TD × division
universe, applies period-correct party/constituency histories, treats formal
abstention as recorded participation, and removes the identified presiding member
from ordinary division opportunities unless that member has a recorded vote
(casting-vote case).

It writes review evidence only. It does not publish or mutate production data.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3
import pandas as pd

from extract.oireachtas.batch import (
    PRODUCTION_POINTER_KEY,
    batch_key_for_production_key,
    batch_manifest_key,
    read_json_required,
)
from political_metrics.audit import history_coverage
from political_metrics.calculators.votes import (
    constituency_vote_participation,
    eligible_division_pairs,
    member_vote_participation,
    party_vote_metrics,
)
from political_metrics.temporal_joins import attach_event_constituency, attach_event_party

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
OUT_DIR = Path(os.getenv("DIRECTOR_VOTE_AUDIT_DIR", "artifacts/director-recorded-voting-participation"))

TABLE_KEYS = {
    "silver_divisions": "processed/oireachtas_unified/latest/csv/silver_divisions.csv",
    "silver_division_tallies": "processed/oireachtas_unified/latest/csv/silver_division_tallies.csv",
    "silver_member_votes": "processed/oireachtas_unified/latest/csv/silver_member_votes.csv",
    "silver_member_memberships": "processed/oireachtas_unified/latest/csv/silver_member_memberships.csv",
    "silver_member_parties": "processed/oireachtas_unified/latest/csv/silver_member_parties.csv",
    "silver_member_constituencies": "processed/oireachtas_unified/latest/csv/silver_member_constituencies.csv",
    "silver_member_offices": "processed/oireachtas_unified/latest/csv/silver_member_offices.csv",
    "silver_members": "processed/oireachtas_unified/latest/csv/silver_members.csv",
    "silver_speeches": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
}

ALLOWED_VOTE_CODES = {"ta", "nil", "staon"}
PRESIDING_RE = re.compile(r"(ceann\s+comhairle|leas[-\s]?cheann\s+comhairle|cathaoirleach|acting\s+chair)", re.I)
PAIRING_LEAVE_RE = re.compile(r"(pair|pairing|leave|maternity|paternity|parental|illness|absence)", re.I)


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in result.columns:
        if result[col].dtype == object:
            result[col] = result[col].fillna("").astype(str).str.strip()
    return result


def _read_batch_csv(s3, *, batch_id: str, logical_key: str) -> tuple[pd.DataFrame, str]:
    key = batch_key_for_production_key(logical_key, batch_id)
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False, na_values=[""]), key


def _iso(value) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _history_report(frame, *, dataset, entity_col, start_col, end_col, details):
    return history_coverage(
        frame,
        dataset=dataset,
        entity_col=entity_col,
        start_col=start_col,
        end_col=end_col,
        detail_columns=details,
    ).as_dict()


def _period_rows(frame: pd.DataFrame, date_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    return frame.loc[dates.between(start, end, inclusive="both")].copy()


def _presiding_lookup(divisions: pd.DataFrame, speeches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Identify the last named chair/presiding member before the end of each division section.

    Primary evidence is a chair-labelled speech in the division's own debate section.
    Fallback evidence is the latest chair-labelled speech earlier in the same debate,
    bounded by the final speech order in the division section. Ambiguities are retained
    in diagnostics rather than silently hidden.
    """
    speech = speeches.copy()
    speech["speech_order_num"] = pd.to_numeric(speech.get("speech_order"), errors="coerce")
    speech["speaker_name"] = speech.get("speaker_name", "").fillna("").astype(str)
    speech["speaker_member_code"] = speech.get("speaker_member_code", "").fillna("").astype(str)
    speech["is_presiding_label"] = speech["speaker_name"].str.contains(PRESIDING_RE, na=False)

    section_max = (
        speech[speech["debate_section_id"].fillna("").astype(str).ne("")]
        .groupby("debate_section_id", as_index=False)["speech_order_num"]
        .max()
        .rename(columns={"speech_order_num": "section_max_speech_order"})
    )
    scoped = divisions[["division_id", "debate_id", "debate_section_id", "division_date"]].merge(
        section_max, on="debate_section_id", how="left", validate="many_to_one"
    )

    chair = speech[
        speech["is_presiding_label"]
        & speech["speaker_member_code"].ne("")
        & speech["debate_id"].fillna("").astype(str).ne("")
    ].copy()

    rows = []
    multi_candidate = []
    role_names: set[str] = set()
    for div in scoped.itertuples(index=False):
        candidates = chair[chair["debate_id"] == div.debate_id].copy()
        if pd.notna(div.section_max_speech_order):
            candidates = candidates[candidates["speech_order_num"] <= div.section_max_speech_order]
        same_section = candidates[candidates["debate_section_id"] == div.debate_section_id].copy()
        pool = same_section if not same_section.empty else candidates
        if pool.empty:
            rows.append({
                "division_id": div.division_id,
                "presiding_member_code": "",
                "presiding_speaker_name": "",
                "presiding_evidence": "unresolved",
            })
            continue
        pool = pool.sort_values(["speech_order_num", "speaker_member_code"])
        distinct = sorted(set(pool["speaker_member_code"].astype(str)) - {""})
        if len(distinct) > 1:
            multi_candidate.append({"division_id": div.division_id, "candidate_member_codes": distinct})
        chosen = pool.iloc[-1]
        role_names.add(str(chosen["speaker_name"]))
        rows.append({
            "division_id": div.division_id,
            "presiding_member_code": str(chosen["speaker_member_code"]),
            "presiding_speaker_name": str(chosen["speaker_name"]),
            "presiding_evidence": "same_section_last_chair_speech" if not same_section.empty else "same_debate_prior_chair_speech",
        })

    lookup = pd.DataFrame(rows)
    diagnostics = {
        "resolved_divisions": int(lookup["presiding_member_code"].astype(str).ne("").sum()),
        "unresolved_divisions": int(lookup["presiding_member_code"].astype(str).eq("").sum()),
        "multi_candidate_divisions": multi_candidate[:50],
        "observed_presiding_speaker_labels": sorted(role_names),
    }
    return lookup, diagnostics


def _named(frame: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    names = members[["member_code", "display_name", "full_name"]].drop_duplicates("member_code").copy()
    result = frame.merge(names, on="member_code", how="left")
    result["member_name"] = result["display_name"].where(result["display_name"].fillna("").astype(str).ne(""), result["full_name"])
    result["member_name"] = result["member_name"].fillna(result["member_code"])
    return result.drop(columns=["display_name", "full_name"])


def _boundary_cases(history: pd.DataFrame, *, start_col: str, end_col: str, period_start: pd.Timestamp, period_end: pd.Timestamp, members: pd.DataFrame) -> list[dict]:
    data = history.copy()
    starts = pd.to_datetime(data[start_col], errors="coerce").dt.normalize()
    ends = pd.to_datetime(data[end_col], errors="coerce").dt.normalize()
    mask = starts.between(period_start, period_end, inclusive="both") | ends.between(period_start, period_end, inclusive="both")
    cols = [c for c in ["member_code", start_col, end_col, "party_name", "constituency_name", "chamber", "house_no"] if c in data.columns]
    cases = _named(data.loc[mask, cols].drop_duplicates(), members)
    return cases.fillna("").to_dict("records")[:100]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ca-central-1")))

    pointer_before = read_json_required(s3, bucket=BUCKET, key=PRODUCTION_POINTER_KEY)
    batch_id = str(pointer_before.get("batch_id") or "").strip()
    if not batch_id:
        raise RuntimeError(f"production pointer is not an immutable batch pointer: {pointer_before}")
    manifest = read_json_required(s3, bucket=BUCKET, key=batch_manifest_key(batch_id))
    if manifest.get("status") != "validated":
        raise RuntimeError(f"production batch {batch_id} is not validated")

    frames = {}
    resolved_keys = {}
    for name, logical in TABLE_KEYS.items():
        frames[name], resolved_keys[name] = _read_batch_csv(s3, batch_id=batch_id, logical_key=logical)
        frames[name] = _clean(frames[name])

    pointer_after = read_json_required(s3, bucket=BUCKET, key=PRODUCTION_POINTER_KEY)
    pointer_stable = pointer_after.get("batch_id") == batch_id

    divisions = frames["silver_divisions"]
    divisions = divisions[divisions["chamber"].str.lower().eq("dail")].copy()
    divisions["division_date_dt"] = pd.to_datetime(divisions["division_date"], errors="coerce").dt.normalize()
    if divisions["division_date_dt"].isna().any():
        raise RuntimeError("Dáil divisions contain invalid division_date values")
    latest_date = divisions["division_date_dt"].max()
    period_end = latest_date
    period_start = period_end - pd.DateOffset(months=6)
    period_divisions = divisions[divisions["division_date_dt"].between(period_start, period_end, inclusive="both")].copy()
    period_ids = set(period_divisions["division_id"])

    tallies = frames["silver_division_tallies"]
    tallies = tallies[tallies["division_id"].isin(period_ids)].copy()
    tallies["member_count_num"] = pd.to_numeric(tallies["member_count"], errors="coerce")
    votes = frames["silver_member_votes"]
    votes = votes[votes["division_id"].isin(period_ids)].copy()
    votes["division_date_dt"] = pd.to_datetime(votes["division_date"], errors="coerce").dt.normalize()

    duplicate_vote_pairs = votes.duplicated(["division_id", "member_code"], keep=False)
    duplicate_vote_pair_count = int(duplicate_vote_pairs.sum())
    invalid_vote_codes = sorted(set(votes["vote_code"].dropna().astype(str)) - ALLOWED_VOTE_CODES)
    invalid_tally_codes = sorted(set(tallies["vote_code"].dropna().astype(str)) - ALLOWED_VOTE_CODES)

    vote_counts = votes.groupby(["division_id", "vote_code"]).size().rename("vote_rows").reset_index()
    tally_counts = tallies.groupby(["division_id", "vote_code"], as_index=False)["member_count_num"].sum()
    by_code = tally_counts.merge(vote_counts, on=["division_id", "vote_code"], how="outer").fillna(0)
    by_code["matches"] = by_code["member_count_num"].astype(int) == by_code["vote_rows"].astype(int)
    division_reconcile = by_code.groupby("division_id", as_index=False).agg(
        tally_rows=("member_count_num", "sum"),
        vote_rows=("vote_rows", "sum"),
        all_vote_codes_match=("matches", "all"),
    )
    duplicate_divisions = set(votes.loc[duplicate_vote_pairs, "division_id"])
    division_reconcile["has_duplicate_member_vote"] = division_reconcile["division_id"].isin(duplicate_divisions)
    division_reconcile["complete_member_vote_coverage"] = (
        division_reconcile["all_vote_codes_match"]
        & ~division_reconcile["has_duplicate_member_vote"]
        & division_reconcile["tally_rows"].eq(division_reconcile["vote_rows"])
    )
    incomplete_ids = set(division_reconcile.loc[~division_reconcile["complete_member_vote_coverage"], "division_id"])
    analysis_divisions = period_divisions[~period_divisions["division_id"].isin(incomplete_ids)].copy()
    analysis_ids = set(analysis_divisions["division_id"])
    analysis_votes = votes[votes["division_id"].isin(analysis_ids)].copy()

    memberships = frames["silver_member_memberships"]
    memberships = memberships[memberships["chamber"].str.lower().eq("dail")].copy()
    member_parties = frames["silver_member_parties"]
    member_constituencies = frames["silver_member_constituencies"]
    members = frames["silver_members"]

    histories = {
        "membership": _history_report(memberships, dataset="silver_member_memberships", entity_col="member_code", start_col="membership_start", end_col="membership_end", details=["membership_id", "house_no", "chamber"]),
        "party": _history_report(member_parties, dataset="silver_member_parties", entity_col="member_code", start_col="party_start", end_col="party_end", details=["party_uri", "party_name", "membership_id"]),
        "constituency": _history_report(member_constituencies, dataset="silver_member_constituencies", entity_col="member_code", start_col="represent_start", end_col="represent_end", details=["constituency_uri", "constituency_name", "membership_id"]),
    }

    eligible_raw = eligible_division_pairs(memberships, analysis_divisions[["division_id", "division_date"]])
    eligible_raw["division_date"] = pd.to_datetime(eligible_raw["division_date"], errors="coerce").dt.normalize()

    presiding, presiding_diag = _presiding_lookup(analysis_divisions, frames["silver_speeches"])
    eligible = eligible_raw.merge(presiding[["division_id", "presiding_member_code", "presiding_evidence"]], on="division_id", how="left", validate="many_to_one")
    vote_pair_index = set(zip(analysis_votes["division_id"], analysis_votes["member_code"]))
    eligible["is_presiding_member"] = eligible["member_code"].eq(eligible["presiding_member_code"])
    eligible["has_recorded_vote"] = [
        (division_id, member_code) in vote_pair_index
        for division_id, member_code in zip(eligible["division_id"], eligible["member_code"])
    ]
    excluded_presiding = eligible[eligible["is_presiding_member"] & ~eligible["has_recorded_vote"]].copy()
    eligible = eligible[~(eligible["is_presiding_member"] & ~eligible["has_recorded_vote"])].copy()

    eligible_key = set(zip(eligible["division_id"], eligible["member_code"]))
    vote_not_eligible = analysis_votes[
        ~pd.Series(list(zip(analysis_votes["division_id"], analysis_votes["member_code"])), index=analysis_votes.index).isin(eligible_key)
    ].copy()

    eligible["event_date"] = eligible["division_date"]
    eligible_party = attach_event_party(eligible, member_parties, event_date_col="event_date")
    eligible_const = attach_event_constituency(eligible, member_constituencies, event_date_col="event_date")
    party_unmatched = int(eligible_party["party_uri"].isna().sum())
    constituency_unmatched = int(eligible_const["constituency_uri"].isna().sum())

    analysis_votes["event_date"] = analysis_votes["division_date"]
    votes_party = attach_event_party(analysis_votes, member_parties, event_date_col="event_date")
    votes_const = attach_event_constituency(analysis_votes, member_constituencies, event_date_col="event_date")
    vote_party_unmatched = int(votes_party["party_uri"].isna().sum())
    vote_const_unmatched = int(votes_const["constituency_uri"].isna().sum())

    td = member_vote_participation(analysis_votes, eligible)
    td = td.rename(columns={"eligible_division_count": "eligible_division_opportunities", "votes_cast_count": "recorded_participation_opportunities"})
    td = _named(td, members)
    td["recorded_participation_pct"] = (td["vote_participation_pct"].astype(float) * 100).round(1)
    td = td[["member_name", "member_code", "recorded_participation_opportunities", "eligible_division_opportunities", "recorded_participation_pct"]].sort_values(["member_name", "member_code"])

    party = party_vote_metrics(analysis_votes, eligible, member_parties)
    party_names = member_parties[["party_uri", "party_name"]].dropna().drop_duplicates("party_uri", keep="last")
    party = party.merge(party_names, on="party_uri", how="left")
    party["recorded_participation_pct"] = (party["vote_participation_pct"].astype(float) * 100).round(1)
    party = party[["party_name", "party_uri", "recorded_member_votes", "eligible_member_divisions", "recorded_participation_pct"]].rename(columns={"recorded_member_votes": "recorded_participation_opportunities", "eligible_member_divisions": "eligible_division_opportunities"}).sort_values(["party_name", "party_uri"])

    constituency = constituency_vote_participation(analysis_votes, eligible, member_constituencies)
    const_names = member_constituencies[["constituency_uri", "constituency_name"]].dropna().drop_duplicates("constituency_uri", keep="last")
    constituency = constituency.merge(const_names, on="constituency_uri", how="left")
    constituency["recorded_participation_pct"] = (constituency["vote_participation_pct"].astype(float) * 100).round(1)
    constituency = constituency[["constituency_name", "constituency_uri", "recorded_member_votes", "eligible_member_divisions", "recorded_participation_pct"]].rename(columns={"recorded_member_votes": "recorded_participation_opportunities", "eligible_member_divisions": "eligible_division_opportunities"}).sort_values(["constituency_name", "constituency_uri"])

    member_numerator = int(td["recorded_participation_opportunities"].sum())
    member_denominator = int(td["eligible_division_opportunities"].sum())
    party_numerator = int(party["recorded_participation_opportunities"].sum())
    party_denominator = int(party["eligible_division_opportunities"].sum())
    const_numerator = int(constituency["recorded_participation_opportunities"].sum())
    const_denominator = int(constituency["eligible_division_opportunities"].sum())

    offices = frames["silver_member_offices"]
    office_mask = offices.get("office_name", pd.Series(index=offices.index, dtype=str)).fillna("").str.contains("Ceann Comhairle", case=False, na=False)
    ceann_offices = _named(offices.loc[office_mask, [c for c in ["member_code", "office_name", "office_start", "office_end", "is_current"] if c in offices.columns]].drop_duplicates(), members)

    abstention_rows = int(analysis_votes["vote_code"].eq("staon").sum())
    abstention_divisions = int(analysis_votes.loc[analysis_votes["vote_code"].eq("staon"), "division_id"].nunique())

    manifest_keyword_hits = []
    for table in manifest.get("tables", []):
        columns = table.get("schema_columns") or []
        text = " ".join([str(table.get("table") or "")] + [str(c) for c in columns])
        if PAIRING_LEAVE_RE.search(text):
            manifest_keyword_hits.append({"table": table.get("table"), "matching_columns": [c for c in columns if PAIRING_LEAVE_RE.search(str(c))]})

    entered_left_cases = _boundary_cases(memberships, start_col="membership_start", end_col="membership_end", period_start=period_start, period_end=period_end, members=members)
    party_change_cases = _boundary_cases(member_parties, start_col="party_start", end_col="party_end", period_start=period_start, period_end=period_end, members=members)

    checks = {
        "production_pointer_stable_during_read": pointer_stable,
        "production_batch_manifest_validated": manifest.get("status") == "validated",
        "period_contains_divisions": len(period_divisions) > 0,
        "td_division_pairs_unique": not eligible.duplicated(["member_code", "division_id"]).any(),
        "member_vote_pairs_unique": duplicate_vote_pair_count == 0,
        "all_vote_codes_recognised": not invalid_vote_codes and not invalid_tally_codes,
        "all_divisions_have_complete_member_vote_coverage": len(incomplete_ids) == 0,
        "all_recorded_votes_are_eligible_after_presiding_rule": len(vote_not_eligible) == 0,
        "membership_history_unambiguous": not histories["membership"]["validation_errors"],
        "party_history_unambiguous": not histories["party"]["validation_errors"],
        "constituency_history_unambiguous": not histories["constituency"]["validation_errors"],
        "all_eligible_opportunities_have_party_attribution": party_unmatched == 0,
        "all_eligible_opportunities_have_constituency_attribution": constituency_unmatched == 0,
        "all_recorded_votes_have_party_attribution": vote_party_unmatched == 0,
        "all_recorded_votes_have_constituency_attribution": vote_const_unmatched == 0,
        "presiding_member_resolved_for_every_analysis_division": presiding_diag["unresolved_divisions"] == 0,
        "party_aggregate_reconciles_to_member_universe": party_numerator == member_numerator and party_denominator == member_denominator,
        "constituency_aggregate_reconciles_to_member_universe": const_numerator == member_numerator and const_denominator == member_denominator,
    }

    ready = all(checks.values())
    report = {
        "audit_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_batch_id": batch_id,
        "production_pointer": pointer_before,
        "resolved_source_keys": resolved_keys,
        "production_division_date_range": {
            "min": _iso(divisions["division_date_dt"].min()),
            "max": _iso(divisions["division_date_dt"].max()),
        },
        "resolved_period": {"start": _iso(period_start), "end": _iso(period_end), "rule": "latest Dáil division date minus six calendar months, inclusive"},
        "division_counts": {
            "in_period": int(len(period_divisions)),
            "included_after_completeness_check": int(len(analysis_divisions)),
            "excluded_for_incomplete_member_vote_coverage": int(len(incomplete_ids)),
            "excluded_division_ids": sorted(incomplete_ids),
        },
        "member_vote_completeness": {
            "duplicate_member_division_rows": duplicate_vote_pair_count,
            "invalid_member_vote_codes": invalid_vote_codes,
            "invalid_tally_vote_codes": invalid_tally_codes,
            "division_reconciliation_failures": division_reconcile.loc[~division_reconcile["complete_member_vote_coverage"]].fillna("").to_dict("records"),
        },
        "abstentions": {
            "representation": "vote_code=staon / formal registered abstention",
            "recorded_abstention_rows": abstention_rows,
            "divisions_with_recorded_abstention": abstention_divisions,
            "counts_as_recorded_participation": True,
        },
        "histories": histories,
        "temporal_attribution": {
            "eligible_party_unmatched_rows": party_unmatched,
            "eligible_constituency_unmatched_rows": constituency_unmatched,
            "vote_party_unmatched_rows": vote_party_unmatched,
            "vote_constituency_unmatched_rows": vote_const_unmatched,
            "entered_or_left_membership_cases": entered_left_cases,
            "party_change_cases": party_change_cases,
        },
        "presiding_member": {
            "rule": "exclude the identified presiding member from an ordinary eligible opportunity unless that member has a recorded vote in the division (casting-vote case)",
            "excluded_ordinary_presiding_opportunities": int(len(excluded_presiding)),
            "identified_ceann_comhairle_office_rows": ceann_offices.fillna("").to_dict("records"),
            **presiding_diag,
        },
        "pairing_or_statutory_leave": {
            "production_manifest_keyword_hits": manifest_keyword_hits,
            "available_as_canonical_adjustment": bool(manifest_keyword_hits),
            "treatment": "No denominator adjustment unless a canonical production field/table is present; external or LLM-inferred absence reasons are not used.",
        },
        "reconciliation": {
            "member_recorded_participation_opportunities": member_numerator,
            "member_eligible_division_opportunities": member_denominator,
            "party_recorded_participation_opportunities": party_numerator,
            "party_eligible_division_opportunities": party_denominator,
            "constituency_recorded_participation_opportunities": const_numerator,
            "constituency_eligible_division_opportunities": const_denominator,
            "recorded_votes_not_in_final_eligible_universe": int(len(vote_not_eligible)),
        },
        "checks": checks,
        "ready_for_content_development": ready,
        "editorial_caveat": "Recorded voting participation does not by itself measure a TD's attendance, workload, effectiveness, or overall job performance.",
    }

    td.to_csv(OUT_DIR / "td_participation.csv", index=False)
    party.to_csv(OUT_DIR / "party_participation.csv", index=False)
    constituency.to_csv(OUT_DIR / "constituency_participation.csv", index=False)
    division_reconcile.to_csv(OUT_DIR / "division_reconciliation.csv", index=False)
    presiding.to_csv(OUT_DIR / "presiding_member_evidence.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    summary = [
        "# Recorded Dáil voting participation audit",
        "",
        f"**Result: {'READY FOR CONTENT DEVELOPMENT' if ready else 'NOT READY'}**",
        "",
        f"Production batch: `{batch_id}`",
        f"Resolved period: **{_iso(period_start)} to {_iso(period_end)}**",
        f"Divisions in period: **{len(period_divisions)}**",
        f"Divisions excluded for incomplete member-vote coverage: **{len(incomplete_ids)}**",
        f"Final eligible TD × division opportunities: **{member_denominator:,}**",
        f"Recorded participation opportunities: **{member_numerator:,}**",
        f"Formal recorded abstentions (`staon`): **{abstention_rows:,}**",
        f"Ordinary presiding-member opportunities removed: **{len(excluded_presiding):,}**",
        "",
        "## Validation checks",
        "",
    ]
    summary.extend([f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}" for name, passed in checks.items()])
    summary.extend([
        "",
        "## Pairing / leave",
        "",
        ("Canonical production fields matching pairing/leave concepts were found; inspect report.json before using them." if manifest_keyword_hits else "No canonical production pairing/statutory-leave field was found in the promoted batch manifest, so no pairing/leave adjustment is made."),
        "",
        "## Interpretation",
        "",
        "Recorded voting participation does not by itself measure a TD's attendance, workload, effectiveness, or overall job performance.",
        "",
    ])
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
