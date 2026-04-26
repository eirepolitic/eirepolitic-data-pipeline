from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


DEFAULT_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
DEFAULT_REGION = os.getenv("AWS_REGION", "ca-central-1")
DEFAULT_URL = os.getenv("VOTES_API_URL", "https://api.oireachtas.ie/v1/votes")
DEFAULT_CHAMBER = os.getenv("VOTES_CHAMBER", "dail")
DEFAULT_HOUSE_NO = os.getenv("VOTES_HOUSE_NO", "34")
DEFAULT_DATE_START = os.getenv("VOTES_DATE_START", "2025-01-01")
DEFAULT_DATE_END = os.getenv("VOTES_DATE_END", "2025-12-31")
DEFAULT_LIMIT = int(os.getenv("VOTES_LIMIT", "10000"))
DEFAULT_TIMEOUT = int(os.getenv("VOTES_TIMEOUT", "60"))
DEFAULT_OUTPUT_CSV_KEY = os.getenv("VOTES_OUTPUT_CSV_KEY", "processed/votes/dail_vote_member_records.csv")
DEFAULT_OUTPUT_PARQUET_KEY = os.getenv(
    "VOTES_OUTPUT_PARQUET_KEY", "processed/votes/parquets/dail_vote_member_records.parquet"
)
DEFAULT_DIVISIONS_CSV_KEY = os.getenv("VOTES_DIVISIONS_CSV_KEY", "processed/votes/dail_vote_divisions.csv")
DEFAULT_DIVISIONS_PARQUET_KEY = os.getenv(
    "VOTES_DIVISIONS_PARQUET_KEY", "processed/votes/parquets/dail_vote_divisions.parquet"
)


@dataclass
class Config:
    bucket: str
    region: str
    api_url: str
    chamber: str
    house_no: str
    date_start: str
    date_end: str
    limit: int
    timeout: int
    output_csv_key: str
    output_parquet_key: str
    divisions_csv_key: str
    divisions_parquet_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-start", default=DEFAULT_DATE_START)
    parser.add_argument("--date-end", default=DEFAULT_DATE_END)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        bucket=DEFAULT_BUCKET,
        region=DEFAULT_REGION,
        api_url=DEFAULT_URL,
        chamber=DEFAULT_CHAMBER,
        house_no=DEFAULT_HOUSE_NO,
        date_start=args.date_start,
        date_end=args.date_end,
        limit=args.limit,
        timeout=DEFAULT_TIMEOUT,
        output_csv_key=DEFAULT_OUTPUT_CSV_KEY,
        output_parquet_key=DEFAULT_OUTPUT_PARQUET_KEY,
        divisions_csv_key=DEFAULT_DIVISIONS_CSV_KEY,
        divisions_parquet_key=DEFAULT_DIVISIONS_PARQUET_KEY,
    )


def fetch_votes_payload(cfg: Config) -> Dict[str, Any]:
    response = requests.get(
        cfg.api_url,
        params={
            "chamber": cfg.chamber,
            "date_start": cfg.date_start,
            "date_end": cfg.date_end,
            "house_no": cfg.house_no,
            "limit": cfg.limit,
        },
        timeout=cfg.timeout,
    )
    response.raise_for_status()
    return response.json()


def text_or_none(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def flatten_member_vote_rows(payload: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    divisions: List[Dict[str, Any]] = []
    member_votes: List[Dict[str, Any]] = []

    for item in payload.get("results", []):
        context_date = text_or_none(item.get("contextDate"))
        division = item.get("division") or {}
        vote_id = text_or_none(division.get("voteId"))
        vote_date = context_date or text_or_none(division.get("date"))
        debate = division.get("debate") or {}
        house = division.get("house") or {}
        subject = division.get("subject") or {}
        tallies = division.get("tallies") or {}

        division_row = {
            "voteId": vote_id,
            "date": vote_date,
            "subject": text_or_none(subject.get("showAs")),
            "outcome": text_or_none(division.get("outcome")),
            "debateShowAs": text_or_none(debate.get("showAs")),
            "debateSection": text_or_none(debate.get("debateSection")),
            "committeeCode": text_or_none(house.get("committeeCode")) or "",
            "unique_vote_id": f"{vote_id}_{vote_date}" if vote_id and vote_date else None,
        }
        divisions.append(division_row)

        for tally_key in ["taVotes", "nilVotes", "staonVotes"]:
            tally = tallies.get(tally_key) or {}
            vote_label = text_or_none(tally.get("showAs"))
            for member_wrapper in tally.get("members", []) or []:
                member = (member_wrapper or {}).get("member") or {}
                member_name = text_or_none(member.get("showAs"))
                member_code = text_or_none(member.get("memberCode"))
                if not member_name or not member_code:
                    continue
                member_votes.append(
                    {
                        "voteId": vote_id,
                        "date": vote_date,
                        "subject": text_or_none(subject.get("showAs")),
                        "outcome": text_or_none(division.get("outcome")),
                        "memberCode": member_code,
                        "memberName": member_name,
                        "vote": vote_label,
                        "debateShowAs": text_or_none(debate.get("showAs")),
                        "debateSection": text_or_none(debate.get("debateSection")),
                        "committeeCode": text_or_none(house.get("committeeCode")) or "",
                        "unique_vote_id": f"{vote_id}_{vote_date}" if vote_id and vote_date else None,
                    }
                )

    df_divisions = pd.DataFrame(divisions)
    if not df_divisions.empty:
        df_divisions = df_divisions[df_divisions["committeeCode"].fillna("") == ""].copy()
        df_divisions["date"] = pd.to_datetime(df_divisions["date"], errors="coerce").dt.date
        df_divisions = df_divisions.drop_duplicates(subset=["unique_vote_id"]).sort_values(
            by=["date", "voteId"], na_position="last"
        )

    df_member_votes = pd.DataFrame(member_votes)
    if not df_member_votes.empty:
        df_member_votes = df_member_votes[df_member_votes["committeeCode"].fillna("") == ""].copy()
        df_member_votes["date"] = pd.to_datetime(df_member_votes["date"], errors="coerce").dt.date
        df_member_votes = df_member_votes.sort_values(
            by=["date", "voteId", "memberName"], na_position="last"
        )

    return df_divisions, df_member_votes


def write_csv_to_s3(s3: Any, *, bucket: str, key: str, df: pd.DataFrame) -> None:
    body = df.to_csv(index=False).encode("utf-8-sig")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/csv")


def write_parquet_to_s3(s3: Any, *, bucket: str, key: str, df: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType="application/x-parquet")


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    payload = fetch_votes_payload(cfg)
    df_divisions, df_member_votes = flatten_member_vote_rows(payload)

    s3 = boto3.client("s3", region_name=cfg.region)
    write_csv_to_s3(s3, bucket=cfg.bucket, key=cfg.divisions_csv_key, df=df_divisions)
    write_parquet_to_s3(s3, bucket=cfg.bucket, key=cfg.divisions_parquet_key, df=df_divisions)
    write_csv_to_s3(s3, bucket=cfg.bucket, key=cfg.output_csv_key, df=df_member_votes)
    write_parquet_to_s3(s3, bucket=cfg.bucket, key=cfg.output_parquet_key, df=df_member_votes)

    print(json.dumps(
        {
            "division_rows": int(len(df_divisions)),
            "member_vote_rows": int(len(df_member_votes)),
            "date_start": cfg.date_start,
            "date_end": cfg.date_end,
            "divisions_csv": f"s3://{cfg.bucket}/{cfg.divisions_csv_key}",
            "member_votes_csv": f"s3://{cfg.bucket}/{cfg.output_csv_key}",
        },
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
