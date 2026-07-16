from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.batch import current_batch_id
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, get_bytes, make_s3_client
from process.instagram_render_post import normalize_name


TARGET_YEAR = int(os.getenv("TARGET_YEAR", str(pd.Timestamp.utcnow().year - 1)))
BUCKET = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
REGION = os.getenv("AWS_REGION", DEFAULT_REGION)
MEMBERS_KEY = os.getenv(
    "MEMBERS_INPUT_KEY",
    "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv",
)
VOTES_KEY = os.getenv(
    "MEMBER_VOTES_INPUT_KEY",
    "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv",
)
PHOTOS_KEY = os.getenv(
    "MEMBER_PHOTOS_INPUT_KEY",
    "processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv",
)
DEBATES_KEY = os.getenv(
    "DEBATE_ISSUES_INPUT_KEY",
    "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv",
)


def metric(name: str) -> str:
    return f"{name}_{TARGET_YEAR}"


def read_csv(s3: Any, key: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(get_bytes(s3, bucket=BUCKET, key=key)), dtype=str, keep_default_na=False)


def output_keys() -> tuple[str, str]:
    batch_id = current_batch_id()
    if batch_id:
        root = f"processed/oireachtas_unified/batches/{batch_id}/consumers/member_profile_metrics"
        return f"{root}/member_profile_metrics_{TARGET_YEAR}.csv", f"{root}/member_profile_metrics_{TARGET_YEAR}.parquet"
    return (
        f"processed/members/member_profile_metrics_{TARGET_YEAR}.csv",
        f"processed/members/parquets/member_profile_metrics_{TARGET_YEAR}.parquet",
    )


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def build_metrics(members: pd.DataFrame, votes: pd.DataFrame, photos: pd.DataFrame, debates: pd.DataFrame) -> pd.DataFrame:
    required = {"member_code", "full_name", "constituency", "party"}
    missing = sorted(required - set(members.columns))
    if missing:
        raise ValueError(f"Members input missing required columns: {missing}")

    base = members.copy()
    base["member_key"] = base["full_name"].map(normalize_name)

    photo_code = _first_column(photos, ["member_code", "memberCode"])
    if photo_code and "photo_url" in photos.columns:
        photo_lookup = photos[[photo_code, "photo_url"]].drop_duplicates(subset=[photo_code]).rename(columns={photo_code: "member_code"})
        base = base.merge(photo_lookup, on="member_code", how="left")
    else:
        base["photo_url"] = ""

    speech_count_col = metric("speech_count")
    speech_rank_col = metric("speech_rank")
    top_issue_col = metric("top_issue")
    top_issue_count_col = metric("top_issue_count")
    speech_metrics = pd.DataFrame(columns=["member_code", speech_count_col, speech_rank_col, top_issue_col, top_issue_count_col])

    speaker_col = _first_column(debates, ["member_code", "speaker_member_code", "Speaker Member Code", "memberCode"])
    speaker_name_col = _first_column(debates, ["Speaker Name", "speaker_name", "member_name"])
    issue_col = _first_column(debates, ["PoliticalIssues", "political_issues", "issue", "Issue", "issue_label", "category", "label"])
    date_col = _first_column(debates, ["Debate Date", "date", "speech_date"])
    if issue_col and (speaker_col or speaker_name_col):
        working = debates.copy()
        if date_col:
            parsed = pd.to_datetime(working[date_col], errors="coerce")
            working = working[parsed.dt.year == TARGET_YEAR].copy()
        if speaker_col:
            working["member_code"] = working[speaker_col].astype(str)
        else:
            working["member_key"] = working[speaker_name_col].map(normalize_name)
            lookup = base[["member_code", "member_key"]].drop_duplicates(subset=["member_key"], keep=False)
            working = working.merge(lookup, on="member_key", how="inner")
        working[issue_col] = working[issue_col].fillna("").astype(str).str.strip()
        working = working[(working["member_code"].str.strip() != "") & (working[issue_col].str.upper() != "NONE") & (working[issue_col] != "")]
        if not working.empty:
            counts = working.groupby("member_code").size().rename(speech_count_col).reset_index()
            counts[speech_rank_col] = counts[speech_count_col].rank(method="dense", ascending=False).astype(int)
            issues = (
                working.groupby(["member_code", issue_col]).size().rename(top_issue_count_col).reset_index()
                .sort_values(["member_code", top_issue_count_col, issue_col], ascending=[True, False, True])
                .drop_duplicates("member_code")
                .rename(columns={issue_col: top_issue_col})
            )
            speech_metrics = counts.merge(issues[["member_code", top_issue_col, top_issue_count_col]], on="member_code", how="left")

    vote_pct_col = metric("vote_participation_pct")
    vote_count_col = metric("distinct_votes_participated")
    all_votes_col = metric("all_distinct_vote_ids")
    vote_metrics = pd.DataFrame(columns=["member_code", vote_pct_col, vote_count_col])
    member_col = _first_column(votes, ["memberCode", "member_code"])
    vote_id_col = _first_column(votes, ["unique_vote_id", "division_id", "vote_id"])
    vote_date_col = _first_column(votes, ["date", "division_date"])
    total_vote_ids = 0
    if member_col and vote_id_col:
        working = votes.copy()
        if vote_date_col:
            parsed = pd.to_datetime(working[vote_date_col], errors="coerce")
            working = working[parsed.dt.year == TARGET_YEAR].copy()
        total_vote_ids = int(working[vote_id_col].replace("", pd.NA).dropna().nunique())
        vote_metrics = (
            working[[member_col, vote_id_col]].replace("", pd.NA).dropna().drop_duplicates()
            .groupby(member_col).size().rename(vote_count_col).reset_index().rename(columns={member_col: "member_code"})
        )
        vote_metrics[vote_pct_col] = (
            (vote_metrics[vote_count_col] / total_vote_ids * 100).round().astype(int)
            if total_vote_ids > 0
            else 0
        )

    output = base.merge(speech_metrics, on="member_code", how="left").merge(vote_metrics, on="member_code", how="left")
    for column in [speech_count_col, speech_rank_col, top_issue_count_col, vote_pct_col, vote_count_col]:
        output[column] = pd.to_numeric(output.get(column), errors="coerce").fillna(0).astype(int)
    output[top_issue_col] = output.get(top_issue_col, "").fillna("").astype(str)
    output[all_votes_col] = total_vote_ids
    keep = [
        "member_code", "full_name", "constituency", "party", "photo_url", top_issue_col,
        top_issue_count_col, vote_pct_col, vote_count_col, all_votes_col, speech_count_col, speech_rank_col,
    ]
    return output.reindex(columns=keep).sort_values([speech_count_col, "full_name"], ascending=[False, True])


def main() -> int:
    s3 = make_s3_client(region_name=REGION)
    output = build_metrics(
        read_csv(s3, MEMBERS_KEY),
        read_csv(s3, VOTES_KEY),
        read_csv(s3, PHOTOS_KEY),
        read_csv(s3, DEBATES_KEY),
    )
    csv_key, parquet_key = output_keys()
    s3.put_object(Bucket=BUCKET, Key=csv_key, Body=output.to_csv(index=False).encode("utf-8-sig"), ContentType="text/csv")
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(output, preserve_index=False), buffer, compression="snappy")
    s3.put_object(Bucket=BUCKET, Key=parquet_key, Body=buffer.getvalue(), ContentType="application/x-parquet")
    print(json.dumps({"rows": len(output), "target_year": TARGET_YEAR, "csv_key": csv_key, "parquet_key": parquet_key}, indent=2))
    return 0 if len(output) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
