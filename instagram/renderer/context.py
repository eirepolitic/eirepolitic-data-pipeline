from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

import pandas as pd

from .data_loader import DatasetBundle
from .util import coalesce_text, normalize_constituency, normalize_name, ordinal_rank, percent_string, safe_int


def pick_issue_column(df_debate: pd.DataFrame) -> Optional[str]:
    for candidate in ["issue", "Issue", "issue_label", "category", "label"]:
        if candidate in df_debate.columns:
            return candidate
    return None


def pick_speaker_column(df_debate: pd.DataFrame) -> Optional[str]:
    for candidate in ["Speaker Name", "speaker_name"]:
        if candidate in df_debate.columns:
            return candidate
    return None


def pick_constituency_image(df_images: pd.DataFrame, constituency_name: str) -> Optional[str]:
    if df_images.empty:
        return None
    key = normalize_constituency(constituency_name)
    for _, row in df_images.iterrows():
        filename = normalize_constituency(row.get("filename", ""))
        if filename == key or key in filename or filename in key:
            return coalesce_text(row.get("url"), row.get("s3_url"))
    return None


def build_member_table(df_members: pd.DataFrame, df_photos: pd.DataFrame, df_summaries: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"member_code", "full_name", "constituency", "party"}
    missing = sorted(required_cols - set(df_members.columns))
    if missing:
        raise RuntimeError(f"Members dataset missing required columns: {missing}")

    base = df_members.copy()
    base["member_key"] = base["full_name"].map(normalize_name)
    base["constituency_key"] = base["constituency"].map(normalize_constituency)

    if not df_photos.empty and "photo_url" in df_photos.columns:
        photos = df_photos.copy()
        if "member_code" in photos.columns:
            base = base.merge(
                photos[["member_code", "photo_url"]].drop_duplicates(subset=["member_code"]),
                on="member_code",
                how="left",
            )
        elif "full_name" in photos.columns:
            photos["member_key"] = photos["full_name"].map(normalize_name)
            base = base.merge(
                photos[["member_key", "photo_url"]].drop_duplicates(subset=["member_key"]),
                on="member_key",
                how="left",
            )

    if not df_summaries.empty and "background" in df_summaries.columns:
        summaries = df_summaries.copy()
        if "member_code" in summaries.columns:
            base = base.merge(
                summaries[["member_code", "background"]].drop_duplicates(subset=["member_code"]),
                on="member_code",
                how="left",
            )
        elif "full_name" in summaries.columns:
            summaries["member_key"] = summaries["full_name"].map(normalize_name)
            base = base.merge(
                summaries[["member_key", "background"]].drop_duplicates(subset=["member_key"]),
                on="member_key",
                how="left",
            )

    return base


def build_issue_records(df_debate: pd.DataFrame, member_lookup: Dict[str, Dict[str, Any]]) -> list[dict[str, Any]]:
    if df_debate.empty:
        return []
    speaker_col = pick_speaker_column(df_debate)
    issue_col = pick_issue_column(df_debate)
    if not speaker_col or not issue_col:
        return []

    records = []
    for _, row in df_debate.iterrows():
        speaker_key = normalize_name(row.get(speaker_col, ""))
        issue = str(row.get(issue_col, "") or "").strip()
        if not speaker_key or not issue or issue.upper() == "NONE":
            continue
        member = member_lookup.get(speaker_key)
        if not member:
            continue
        records.append({
            "member_key": speaker_key,
            "constituency_key": member.get("constituency_key"),
            "issue": issue,
        })
    return records


def issue_counts_from_records(records: list[dict[str, Any]], *, constituency_key: str | None = None, member_key: str | None = None) -> Counter:
    counter: Counter = Counter()
    for rec in records:
        if constituency_key and rec.get("constituency_key") != constituency_key:
            continue
        if member_key and rec.get("member_key") != member_key:
            continue
        counter[rec["issue"]] += 1
    return counter


def build_post_context(spec: Dict[str, Any], bundle: DatasetBundle) -> Dict[str, Any]:
    tables = bundle.tables
    df_members = build_member_table(
        tables["members"],
        tables["member_photos"],
        tables["member_summaries"],
    )

    constituency_name = spec["data"]["constituency"]
    constituency_key = normalize_constituency(constituency_name)

    member_lookup = {
        normalize_name(row.get("full_name")): row.to_dict()
        for _, row in df_members.iterrows()
        if str(row.get("full_name", "")).strip()
    }

    members_in_constituency = df_members[df_members["constituency_key"] == constituency_key].copy()
    if members_in_constituency.empty:
        sample = sorted(set(df_members["constituency"].dropna().astype(str).tolist()))[:20]
        raise RuntimeError(
            f"No members matched constituency '{constituency_name}'. Sample available constituencies: {sample}"
        )

    issue_records = build_issue_records(tables["debate_issues"], member_lookup)
    constituency_issue_counts = issue_counts_from_records(issue_records, constituency_key=constituency_key)
    speech_count_map = Counter(
        rec["member_key"] for rec in issue_records if rec.get("constituency_key") == constituency_key
    )
    members_in_constituency["speech_count"] = members_in_constituency["member_key"].map(lambda x: speech_count_map.get(x, 0))

    requested_member = coalesce_text(spec["data"].get("member_name"))
    selected_member = None
    if requested_member:
        requested_key = normalize_name(requested_member)
        matched = members_in_constituency[members_in_constituency["member_key"] == requested_key]
        if not matched.empty:
            selected_member = matched.iloc[0]

    if selected_member is None:
        members_in_constituency = members_in_constituency.sort_values(
            by=["speech_count", "full_name"],
            ascending=[False, True],
        )
        selected_member = members_in_constituency.iloc[0]

    member_key = selected_member["member_key"]
    member_issue_counts = issue_counts_from_records(issue_records, member_key=member_key)

    top_constituency_issue = constituency_issue_counts.most_common(1)[0][0] if constituency_issue_counts else "No classified issue yet"
    top_member_issue = member_issue_counts.most_common(1)[0][0] if member_issue_counts else "No classified issue yet"

    metrics = spec.get("data", {}).get("metrics", {})
    party_count = int(
        members_in_constituency["party"]
        .fillna("")
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )
    constituency = {
        "name": constituency_name,
        "member_count": int(len(members_in_constituency)),
        "party_count": party_count,
        "speech_count": int(sum(constituency_issue_counts.values())),
        "image_url": pick_constituency_image(tables["constituency_images"], constituency_name),
        "top_issue_label": top_constituency_issue,
        "vote_participation_pct": percent_string(metrics.get("vote_participation_pct")),
        "speech_rank": ordinal_rank(safe_int(metrics.get("constituency_speech_rank") or metrics.get("speech_rank"))),
    }

    member = {
        "full_name": coalesce_text(selected_member.get("full_name")) or "Unknown member",
        "party": coalesce_text(selected_member.get("party")) or "Party unavailable",
        "constituency": coalesce_text(selected_member.get("constituency")) or constituency_name,
        "photo_url": coalesce_text(selected_member.get("photo_url")),
        "background": coalesce_text(selected_member.get("background")) or "Background summary unavailable.",
        "speech_count": safe_int(selected_member.get("speech_count")),
        "top_issue_label": top_member_issue,
        "vote_participation_pct": percent_string(metrics.get("vote_participation_pct")),
        "speech_rank": ordinal_rank(safe_int(metrics.get("speech_rank"))),
        "member_key": member_key,
    }

    issue_rows = [
        {"label": label, "count": count}
        for label, count in constituency_issue_counts.most_common(int(spec["data"].get("issue_limit", 8)))
    ]
    member_issue_rows = [
        {"label": label, "count": count}
        for label, count in member_issue_counts.most_common(int(spec["data"].get("issue_limit", 8)))
    ]

    return {
        "post": spec["post"],
        "branding": spec["branding"],
        "style": spec.get("style", {}),
        "data": spec.get("data", {}),
        "slides": [slide for slide in spec["slides"] if slide.get("enabled", True)],
        "slide_size": spec["post"]["slide_size"],
        "datasets_used": bundle.sources,
        "constituency": constituency,
        "member": member,
        "constituency_issue_rows": issue_rows,
        "member_issue_rows": member_issue_rows,
        "glossary": spec.get("data", {}).get("glossary", {}),
    }
