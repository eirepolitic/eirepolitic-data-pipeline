#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import boto3
import pandas as pd

from extract.oireachtas.batch import resolve_production_key

QUESTIONS_KEY = "processed/oireachtas_unified/latest/csv/silver_questions.csv"
SECTIONS_KEY = "processed/oireachtas_unified/latest/metrics/event/written_question_answer_sections/csv/written_question_answer_sections.csv"
BRIDGE_KEY = "processed/oireachtas_unified/latest/metrics/event/written_question_answer_bridge/csv/written_question_answer_bridge.csv"


def read_s3_csv(s3, bucket: str, logical_key: str) -> pd.DataFrame:
    key = resolve_production_key(s3, bucket=bucket, production_key=logical_key)
    payload = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False, na_values=[""])


def as_bool(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def build_section_frame(questions: pd.DataFrame, sections: pd.DataFrame, bridge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    qcols = ["question_id", "question_text", "question_date", "asked_by_name", "to_minister_or_department", "question_no"]
    linked = bridge[["question_id", "debate_section_id"]].merge(
        questions[qcols], on="question_id", how="left", validate="many_to_one"
    )
    first = linked.groupby("debate_section_id", sort=False).first().reset_index()[
        ["debate_section_id", "question_date", "to_minister_or_department"]
    ]
    question_counts = linked.groupby("debate_section_id").size().rename("linked_question_count").reset_index()
    sec_cols = [
        "debate_section_id", "answer_text", "answer_status", "grouped_answer", "referred_or_direct_reply",
        "embedded_table_count", "section_heading", "source_xml_url",
    ]
    frame = sections[sec_cols].merge(first, on="debate_section_id", how="inner", validate="one_to_one")
    frame = frame.merge(question_counts, on="debate_section_id", how="left", validate="one_to_one")
    frame["grouped_answer"] = as_bool(frame["grouped_answer"])
    frame["referred_or_direct_reply"] = as_bool(frame["referred_or_direct_reply"])
    frame["embedded_table_count"] = pd.to_numeric(frame["embedded_table_count"], errors="coerce").fillna(0).astype(int)
    frame["answer_text"] = frame["answer_text"].fillna("").astype(str)
    frame["answer_chars"] = frame["answer_text"].str.len()
    frame["year"] = frame["question_date"].fillna("").astype(str).str[:4]
    return frame, linked


def choose_gold_set(frame: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    selected: set[str] = set()

    def add(subset: pd.DataFrame, n: int, offset: int) -> None:
        if n <= 0 or subset.empty:
            return
        pool = subset[~subset["debate_section_id"].astype(str).isin(selected)].copy()
        if pool.empty:
            return
        take = pool.sample(n=min(n, len(pool)), random_state=seed + offset)
        selected.update(take["debate_section_id"].astype(str).tolist())

    nonempty = frame[frame["answer_chars"] > 0]
    q10 = float(nonempty["answer_chars"].quantile(0.10)) if not nonempty.empty else 0
    q90 = float(nonempty["answer_chars"].quantile(0.90)) if not nonempty.empty else 0

    add(frame[frame["answer_status"].fillna("").ne("ministerial_reply_present")], 10, 1)
    add(frame[frame["grouped_answer"]], 15, 2)
    add(frame[frame["referred_or_direct_reply"]], 15, 3)
    add(frame[frame["embedded_table_count"] > 0], 10, 4)
    add(frame[frame["answer_chars"] >= q90], 15, 5)
    add(frame[(frame["answer_chars"] > 0) & (frame["answer_chars"] <= q10)], 10, 6)

    # Ensure broad recipient coverage before random fill.
    remaining = frame[~frame["debate_section_id"].astype(str).isin(selected)].copy()
    if not remaining.empty:
        diverse = (
            remaining.sort_values(["to_minister_or_department", "question_date", "debate_section_id"])
            .groupby("to_minister_or_department", dropna=False, sort=True)
            .head(1)
        )
        for sid in diverse["debate_section_id"].astype(str).tolist():
            if len(selected) >= target:
                break
            selected.add(sid)

    remaining = frame[~frame["debate_section_id"].astype(str).isin(selected)].copy()
    if len(selected) < target and not remaining.empty:
        add(remaining, target - len(selected), 99)

    result = frame[frame["debate_section_id"].astype(str).isin(selected)].copy()
    if len(result) > target:
        result = result.sample(n=target, random_state=seed)
    return result.sort_values(["question_date", "debate_section_id"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Create a blind human-review gold-set template for Written PQ semantic classification.")
    p.add_argument("--target-sections", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--output-dir", default="analysis/written_pq_semantic_gold_set")
    args = p.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    questions = read_s3_csv(s3, args.bucket, QUESTIONS_KEY)
    sections = read_s3_csv(s3, args.bucket, SECTIONS_KEY)
    bridge = read_s3_csv(s3, args.bucket, BRIDGE_KEY)
    frame, linked = build_section_frame(questions, sections, bridge)
    gold = choose_gold_set(frame, args.target_sections, args.seed)
    ids = set(gold["debate_section_id"].astype(str))
    qreview = linked[linked["debate_section_id"].astype(str).isin(ids)].copy()

    id_map = {sid: f"WPQG{idx:03d}" for idx, sid in enumerate(gold["debate_section_id"].astype(str), 1)}
    gold.insert(0, "review_id", gold["debate_section_id"].astype(str).map(id_map))
    qreview.insert(0, "review_id", qreview["debate_section_id"].astype(str).map(id_map))

    manifest_cols = [
        "review_id", "debate_section_id", "question_date", "year", "to_minister_or_department", "answer_status",
        "grouped_answer", "referred_or_direct_reply", "embedded_table_count", "linked_question_count", "answer_chars",
        "section_heading", "source_xml_url",
    ]
    manifest = gold[manifest_cols].copy()

    question_review = qreview[[
        "review_id", "debate_section_id", "question_id", "question_no", "question_date", "asked_by_name",
        "to_minister_or_department", "question_text",
    ]].copy()
    question_review["human_topic_tags_json"] = ""
    question_review["human_question_intents_json"] = ""
    question_review["human_missing_topic_notes"] = ""
    question_review["reviewer_notes"] = ""

    answer_review = gold[[
        "review_id", "debate_section_id", "question_date", "to_minister_or_department", "answer_status",
        "grouped_answer", "referred_or_direct_reply", "embedded_table_count", "section_heading", "answer_text", "source_xml_url",
    ]].copy()
    answer_review["human_topic_tags_json"] = ""
    answer_review["human_answer_characteristics_json"] = ""
    answer_review["human_missing_topic_notes"] = ""
    answer_review["reviewer_notes"] = ""

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "gold_set_manifest.csv", index=False)
    question_review.to_csv(out / "question_review.csv", index=False)
    answer_review.to_csv(out / "answer_review.csv", index=False)

    summary = {
        "target_sections": args.target_sections,
        "selected_sections": int(len(gold)),
        "selected_questions": int(len(question_review)),
        "years": {str(k): int(v) for k, v in gold.groupby("year").size().to_dict().items()},
        "unique_recipients": int(gold["to_minister_or_department"].nunique(dropna=True)),
        "grouped_sections": int(gold["grouped_answer"].sum()),
        "referral_sections": int(gold["referred_or_direct_reply"].sum()),
        "non_ministerial_reply_status_sections": int(gold["answer_status"].fillna("").ne("ministerial_reply_present").sum()),
        "sections_with_tables": int((gold["embedded_table_count"] > 0).sum()),
        "answer_chars_min": int(gold["answer_chars"].min()),
        "answer_chars_median": float(gold["answer_chars"].median()),
        "answer_chars_max": int(gold["answer_chars"].max()),
        "blind_to_model_outputs": True,
        "production_changed": False,
        "seed": args.seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
