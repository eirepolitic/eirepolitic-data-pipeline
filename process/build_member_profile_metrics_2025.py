from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Iterable, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from instagram_render_post import normalize_name


DEFAULT_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
DEFAULT_REGION = os.getenv("AWS_REGION", "ca-central-1")
INPUT_MEMBERS_KEY = os.getenv("MEMBERS_INPUT_KEY", "raw/members/oireachtas_members_34th_dail.csv")
INPUT_PHOTOS_KEY = os.getenv("MEMBER_PHOTOS_INPUT_KEY", "processed/members/members_photo_urls.csv")
INPUT_DEBATES_KEY = os.getenv("DEBATE_ISSUES_INPUT_KEY", "processed/debates/debate_speeches_classified.csv")
INPUT_VOTES_KEY = os.getenv("MEMBER_VOTES_INPUT_KEY", "processed/votes/dail_vote_member_records.csv")
OUTPUT_CSV_KEY = os.getenv("MEMBER_PROFILE_METRICS_OUTPUT_CSV_KEY", "processed/members/member_profile_metrics_2025.csv")
OUTPUT_PARQUET_KEY = os.getenv(
    "MEMBER_PROFILE_METRICS_OUTPUT_PARQUET_KEY",
    "processed/members/parquets/member_profile_metrics_2025.parquet",
)
TARGET_YEAR = int(os.getenv("TARGET_YEAR", "2025"))


def read_csv_from_s3(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text))


def write_csv_to_s3(s3: Any, *, bucket: str, key: str, df: pd.DataFrame) -> None:
    body = df.to_csv(index=False).encode("utf-8-sig")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/csv")


def write_parquet_to_s3(s3: Any, *, bucket: str, key: str, df: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType="application/x-parquet")


def pick_issue_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["PoliticalIssues", "political_issues", "issue", "Issue", "issue_label", "category", "label"]:
        if col in df.columns:
            return col
    return None


def pick_speaker_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["Speaker Name", "speaker_name"]:
        if col in df.columns:
            return col
    return None


def build_member_base(df_members: pd.DataFrame, df_photos: pd.DataFrame) -> pd.DataFrame:
    base = df_members.copy()
    base["member_key"] = base["full_name"].map(normalize_name)

    if not df_photos.empty:
        photos = df_photos.copy()
        if "member_code" in photos.columns:
            photos = photos.rename(columns={"member_code": "member_code_photo"})
        if "memberCode" in photos.columns:
            photos = photos.rename(columns={"memberCode": "member_code_photo"})

        if "full_name" in photos.columns:
            photos["member_key"] = photos["full_name"].map(normalize_name)

        if "member_code_photo" in photos.columns:
            base = base.merge(
                photos[["member_code_photo", "photo_url"]].drop_duplicates(subset=["member_code_photo"]),
                left_on="member_code",
                right_on="member_code_photo",
                how="left",
            )
        elif "member_key" in photos.columns:
            base = base.merge(
                photos[["member_key", "photo_url"]].drop_duplicates(subset=["member_key"]),
                on="member_key",
                how="left",
            )

    return base


def compute_speech_metrics(df_members: pd.DataFrame, df_debates: pd.DataFrame) -> pd.DataFrame:
    if df_debates.empty:
        return pd.DataFrame(columns=["member_code", "top_issue_2025", "top_issue_count_2025", "speech_count_2025", "speech_rank_2025"])

    issue_col = pick_issue_column(df_debates)
    speaker_col = pick_speaker_column(df_debates)
    if not issue_col or not speaker_col:
        return pd.DataFrame(columns=["member_code", "top_issue_2025", "top_issue_count_2025", "speech_count_2025", "speech_rank_2025"])

    debates = df_debates.copy()
    date_col = "Debate Date" if "Debate Date" in debates.columns else "date" if "date" in debates.columns else None
    if date_col:
        debates["parsed_date"] = pd.to_datetime(debates[date_col], errors="coerce")
        debates = debates[debates["parsed_date"].dt.year == TARGET_YEAR].copy()

    debates[issue_col] = debates[issue_col].fillna("").astype(str).str.strip()
    debates = debates[debates[issue_col].str.upper() != "NONE"].copy()
    debates["member_key"] = debates[speaker_col].map(normalize_name)

    lookup = df_members[["member_code", "member_key"]].drop_duplicates(subset=["member_key"])
    debates = debates.merge(lookup, on="member_key", how="inner")

    if debates.empty:
        return pd.DataFrame(columns=["member_code", "top_issue_2025", "top_issue_count_2025", "speech_count_2025", "speech_rank_2025"])

    speech_counts = debates.groupby("member_code").size().rename("speech_count_2025").reset_index()
    speech_counts["speech_rank_2025"] = speech_counts["speech_count_2025"].rank(method="dense", ascending=False).astype(int)

    issue_counts = (
        debates.groupby(["member_code", issue_col]).size().rename("issue_count").reset_index().sort_values(
            by=["member_code", "issue_count", issue_col], ascending=[True, False, True]
        )
    )
    top_issue = issue_counts.drop_duplicates(subset=["member_code"]).rename(
        columns={issue_col: "top_issue_2025", "issue_count": "top_issue_count_2025"}
    )[["member_code", "top_issue_2025", "top_issue_count_2025"]]

    return speech_counts.merge(top_issue, on="member_code", how="left")


def compute_vote_metrics(df_votes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df_votes.empty:
        return pd.DataFrame(columns=["member_code", "vote_participation_pct_2025", "distinct_votes_participated_2025"]), 0

    votes = df_votes.copy()
    if "date" in votes.columns:
        votes["parsed_date"] = pd.to_datetime(votes["date"], errors="coerce")
        votes = votes[votes["parsed_date"].dt.year == TARGET_YEAR].copy()

    member_code_col = "memberCode" if "memberCode" in votes.columns else "member_code" if "member_code" in votes.columns else None
    unique_vote_col = "unique_vote_id" if "unique_vote_id" in votes.columns else None
    if not member_code_col or not unique_vote_col:
        return pd.DataFrame(columns=["member_code", "vote_participation_pct_2025", "distinct_votes_participated_2025"]), 0

    all_distinct_vote_ids = votes[unique_vote_col].dropna().astype(str).nunique()

    participated = (
        votes[[member_code_col, unique_vote_col]]
        .dropna()
        .drop_duplicates()
        .groupby(member_code_col)
        .size()
        .rename("distinct_votes_participated_2025")
        .reset_index()
        .rename(columns={member_code_col: "member_code"})
    )
    if all_distinct_vote_ids > 0:
        participated["vote_participation_pct_2025"] = (
            participated["distinct_votes_participated_2025"] / all_distinct_vote_ids * 100
        ).round(0).astype(int)
    else:
        participated["vote_participation_pct_2025"] = 0

    return participated, all_distinct_vote_ids


def main() -> None:
    s3 = boto3.client("s3", region_name=DEFAULT_REGION)

    df_members = read_csv_from_s3(s3, DEFAULT_BUCKET, INPUT_MEMBERS_KEY)
    df_photos = read_csv_from_s3(s3, DEFAULT_BUCKET, INPUT_PHOTOS_KEY)
    df_debates = read_csv_from_s3(s3, DEFAULT_BUCKET, INPUT_DEBATES_KEY)
    df_votes = read_csv_from_s3(s3, DEFAULT_BUCKET, INPUT_VOTES_KEY)

    base = build_member_base(df_members, df_photos)
    speech_metrics = compute_speech_metrics(base, df_debates)
    vote_metrics, total_vote_ids = compute_vote_metrics(df_votes)

    output = base.merge(speech_metrics, on="member_code", how="left").merge(vote_metrics, on="member_code", how="left")
    output["vote_participation_pct_2025"] = output["vote_participation_pct_2025"].fillna(0).astype(int)
    output["distinct_votes_participated_2025"] = output["distinct_votes_participated_2025"].fillna(0).astype(int)
    output["speech_count_2025"] = output["speech_count_2025"].fillna(0).astype(int)
    output["top_issue_count_2025"] = output["top_issue_count_2025"].fillna(0).astype(int)
    output["speech_rank_2025"] = output["speech_rank_2025"].fillna(0).astype(int)
    output["all_distinct_vote_ids_2025"] = total_vote_ids

    keep_cols = [
        "member_code",
        "full_name",
        "constituency",
        "party",
        "photo_url",
        "top_issue_2025",
        "top_issue_count_2025",
        "vote_participation_pct_2025",
        "distinct_votes_participated_2025",
        "all_distinct_vote_ids_2025",
        "speech_count_2025",
        "speech_rank_2025",
    ]
    output = output[keep_cols].sort_values(by=["speech_count_2025", "full_name"], ascending=[False, True])

    write_csv_to_s3(s3, bucket=DEFAULT_BUCKET, key=OUTPUT_CSV_KEY, df=output)
    write_parquet_to_s3(s3, bucket=DEFAULT_BUCKET, key=OUTPUT_PARQUET_KEY, df=output)

    print(json.dumps(
        {
            "rows": int(len(output)),
            "year": TARGET_YEAR,
            "member_metrics_csv": f"s3://{DEFAULT_BUCKET}/{OUTPUT_CSV_KEY}",
            "total_distinct_vote_ids": int(total_vote_ids),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
