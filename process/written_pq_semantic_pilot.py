#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import yaml
from openai import OpenAI

from extract.oireachtas.batch import resolve_production_key

DEFAULT_MODEL = "gpt-5.6-luna"
CONFIG_PATH = Path("configs/political_metrics/written_pq_semantic_pilot.yml")
QUESTIONS_KEY = "processed/oireachtas_unified/latest/csv/silver_questions.csv"
SECTIONS_KEY = "processed/oireachtas_unified/latest/metrics/event/written_question_answer_sections/csv/written_question_answer_sections.csv"
BRIDGE_KEY = "processed/oireachtas_unified/latest/metrics/event/written_question_answer_bridge/csv/written_question_answer_bridge.csv"


def read_s3_csv(s3, bucket: str, logical_key: str) -> pd.DataFrame:
    key = resolve_production_key(s3, bucket=bucket, production_key=logical_key)
    payload = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False, na_values=[""])


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def allowed_topic_tags(config: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for topic in config["broad_topics"]:
        tags.append(topic["id"])
        tags.extend(topic.get("children", []))
    return tags


def schema(config: dict[str, Any]) -> dict[str, Any]:
    topic_tags = allowed_topic_tags(config)
    intents = config["question_intents"]
    characteristics = config["answer_characteristics"]
    entity_types = config["entity_types"]
    tag_array = {"type": "array", "items": {"type": "string", "enum": topic_tags}, "uniqueItems": True}
    proposed = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "tag": {"type": "string"},
                "reason": {"type": "string"},
                "evidence_quote": {"type": "string"},
            },
            "required": ["tag", "reason", "evidence_quote"],
            "additionalProperties": False,
        },
    }
    evidence = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "enum": topic_tags},
                "evidence_quote": {"type": "string"},
            },
            "required": ["tag", "evidence_quote"],
            "additionalProperties": False,
        },
    }
    entities = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": entity_types},
                "evidence_quote": {"type": "string"},
            },
            "required": ["name", "type", "evidence_quote"],
            "additionalProperties": False,
        },
    }
    question_item = {
        "type": "object",
        "properties": {
            "question_id": {"type": "string"},
            "topic_tags": tag_array,
            "question_intents": {"type": "array", "items": {"type": "string", "enum": intents}, "uniqueItems": True},
            "entities": entities,
            "proposed_new_tags": proposed,
            "topic_evidence": evidence,
        },
        "required": ["question_id", "topic_tags", "question_intents", "entities", "proposed_new_tags", "topic_evidence"],
        "additionalProperties": False,
    }
    answer_obj = {
        "type": "object",
        "properties": {
            "topic_tags": tag_array,
            "answer_characteristics": {"type": "array", "items": {"type": "string", "enum": characteristics}, "uniqueItems": True},
            "entities": entities,
            "proposed_new_tags": proposed,
            "topic_evidence": evidence,
        },
        "required": ["topic_tags", "answer_characteristics", "entities", "proposed_new_tags", "topic_evidence"],
        "additionalProperties": False,
    }
    combined_obj = {
        "type": "object",
        "properties": {
            "topic_tags": tag_array,
            "proposed_new_tags": proposed,
            "topic_evidence": evidence,
        },
        "required": ["topic_tags", "proposed_new_tags", "topic_evidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "questions": {"type": "array", "items": question_item},
            "answer": answer_obj,
            "combined_exchange": combined_obj,
        },
        "required": ["questions", "answer", "combined_exchange"],
        "additionalProperties": False,
    }


def prompt(config: dict[str, Any], record: dict[str, Any]) -> list[dict[str, str]]:
    taxonomy = []
    for parent in config["broad_topics"]:
        taxonomy.append(f"{parent['id']}: {parent['label']}")
        taxonomy.extend(f"  - {child}" for child in parent.get("children", []))
    questions = "\n\n".join(
        f"QUESTION_ID: {q['question_id']}\nQUESTION: {q['question_text']}" for q in record["questions"]
    )
    system = (
        "You are performing research-only semantic indexing of Irish Parliamentary Questions. "
        "Classify the supplied question texts, the official answer, and the combined exchange separately. "
        "Use multiple topic tags where genuinely relevant. Prefer the most specific approved tags and include a broad parent only when it adds useful routing value. "
        "Do not infer political support/opposition, truthfulness, quality, evasiveness, effectiveness, motives, or unstated positions. "
        "Known metadata such as department, TD, date and party are supplied separately and must not be invented as semantic tags. "
        "Only use proposed_new_tags when the approved taxonomy genuinely lacks a useful recurring concept. "
        "Evidence quotes must be short verbatim snippets from the supplied text."
    )
    user = (
        "APPROVED TOPIC TAXONOMY:\n" + "\n".join(taxonomy) +
        "\n\nQUESTION INTENTS:\n- " + "\n- ".join(config["question_intents"]) +
        "\n\nANSWER CHARACTERISTICS:\n- " + "\n- ".join(config["answer_characteristics"]) +
        "\n\nSOURCE METADATA (do not recreate as topic tags):\n" +
        json.dumps(record["metadata"], ensure_ascii=False) +
        "\n\n" + questions +
        "\n\nOFFICIAL ANSWER:\n" + record["answer_text"]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def classify(client: OpenAI, config: dict[str, Any], record: dict[str, Any], model: str) -> tuple[dict[str, Any], dict[str, int]]:
    response = client.responses.create(
        model=model,
        input=prompt(config, record),
        reasoning={"effort": "low"},
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "written_pq_semantic_index",
                "strict": True,
                "schema": schema(config),
            },
        },
        max_output_tokens=3000,
        store=False,
    )
    result = json.loads(str(response.output_text or "").strip())
    usage = getattr(response, "usage", None)
    usage_dict = {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return result, usage_dict


def build_exchange_frame(questions: pd.DataFrame, sections: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    qcols = ["question_id", "question_text", "question_date", "asked_by_name", "to_minister_or_department", "question_no"]
    q = questions[qcols].copy()
    b = bridge[["question_id", "debate_section_id"]].copy()
    joined = b.merge(q, on="question_id", how="left", validate="many_to_one")
    qgroups = joined.groupby("debate_section_id", sort=False).apply(
        lambda g: [
            {
                "question_id": str(r.question_id),
                "question_text": str(r.question_text or ""),
                "question_no": str(r.question_no or ""),
                "asked_by_name": str(r.asked_by_name or ""),
            }
            for r in g.itertuples(index=False)
        ],
        include_groups=False,
    ).rename("questions").reset_index()
    first = joined.groupby("debate_section_id", sort=False).first().reset_index()[
        ["debate_section_id", "question_date", "to_minister_or_department"]
    ]
    sec_cols = [
        "debate_section_id", "answer_text", "answer_status", "grouped_answer", "referred_or_direct_reply",
        "section_heading", "embedded_table_count", "respondent_ref", "respondent_role_ref",
    ]
    result = sections[sec_cols].merge(qgroups, on="debate_section_id", how="inner", validate="one_to_one")
    return result.merge(first, on="debate_section_id", how="left", validate="one_to_one")


def stratified_sample(frame: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    work = frame.copy()
    work["answer_length"] = work["answer_text"].fillna("").astype(str).str.len()
    work["answer_length_band"] = pd.qcut(work["answer_length"].rank(method="first"), q=4, labels=["short", "medium", "long", "very_long"])
    work["year"] = work["question_date"].fillna("").astype(str).str[:4]
    work["grouped_answer"] = work["grouped_answer"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
    work["referred_or_direct_reply"] = work["referred_or_direct_reply"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
    work["stratum"] = (
        work["year"].astype(str) + "|" + work["answer_length_band"].astype(str) + "|" +
        work["grouped_answer"].astype(str) + "|" + work["referred_or_direct_reply"].astype(str) + "|" +
        work["answer_status"].fillna("").astype(str)
    )
    groups = list(work.groupby("stratum", sort=True))
    per = max(1, target // max(1, len(groups)))
    parts = [g.sample(n=min(per, len(g)), random_state=seed + i) for i, (_, g) in enumerate(groups)]
    sample = pd.concat(parts, ignore_index=True) if parts else work.head(0)
    if len(sample) < target:
        remaining = work[~work["debate_section_id"].isin(sample["debate_section_id"])].copy()
        if not remaining.empty:
            add = remaining.sample(n=min(target - len(sample), len(remaining)), random_state=seed)
            sample = pd.concat([sample, add], ignore_index=True)
    elif len(sample) > target:
        sample = sample.sample(n=target, random_state=seed)
    return sample.sort_values(["question_date", "debate_section_id"]).reset_index(drop=True)


def validate_result(record: dict[str, Any], result: dict[str, Any], config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_qids = [q["question_id"] for q in record["questions"]]
    actual_qids = [q.get("question_id", "") for q in result.get("questions", [])]
    if sorted(expected_qids) != sorted(actual_qids):
        errors.append("question_id_set_mismatch")
    allowed = set(allowed_topic_tags(config))
    for scope, obj in [("answer", result.get("answer", {})), ("combined", result.get("combined_exchange", {}))]:
        if not set(obj.get("topic_tags", [])).issubset(allowed):
            errors.append(f"invalid_{scope}_topic_tag")
    for q in result.get("questions", []):
        if not set(q.get("topic_tags", [])).issubset(allowed):
            errors.append("invalid_question_topic_tag")
    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--max-calls", type=int, default=0)
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--output-dir", default="analysis/written_pq_semantic_pilot")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    config = load_config()
    s3 = boto3.client("s3", region_name=args.region)
    questions = read_s3_csv(s3, args.bucket, QUESTIONS_KEY)
    sections = read_s3_csv(s3, args.bucket, SECTIONS_KEY)
    bridge = read_s3_csv(s3, args.bucket, BRIDGE_KEY)
    frame = build_exchange_frame(questions, sections, bridge)
    sample = stratified_sample(frame, args.sample_size, args.seed)
    if args.max_calls > 0:
        sample = sample.head(args.max_calls).copy()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    failures = []
    for i, r in enumerate(sample.to_dict("records"), 1):
        questions_payload = r["questions"]
        record = {
            "questions": questions_payload,
            "answer_text": str(r.get("answer_text") or ""),
            "metadata": {
                "debate_section_id": str(r.get("debate_section_id") or ""),
                "question_date": str(r.get("question_date") or ""),
                "recipient": str(r.get("to_minister_or_department") or ""),
                "answer_status": str(r.get("answer_status") or ""),
                "grouped_answer": bool(r.get("grouped_answer")),
                "referred_or_direct_reply": bool(r.get("referred_or_direct_reply")),
                "section_heading": str(r.get("section_heading") or ""),
                "embedded_table_count": str(r.get("embedded_table_count") or "0"),
            },
        }
        try:
            result, usage = classify(client, config, record, args.model)
            errors = validate_result(record, result, config)
            totals = {k: totals[k] + usage[k] for k in totals}
            rows.append({
                "debate_section_id": record["metadata"]["debate_section_id"],
                "question_date": record["metadata"]["question_date"],
                "recipient": record["metadata"]["recipient"],
                "question_count": len(questions_payload),
                "answer_chars": len(record["answer_text"]),
                "classification_json": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                "validation_errors": ";".join(errors),
                **usage,
            })
        except Exception as exc:
            failures.append({"debate_section_id": record["metadata"]["debate_section_id"], "error": f"{type(exc).__name__}: {exc}"})
        if i % 25 == 0:
            print(json.dumps({"completed": i, "target": len(sample), "failures": len(failures), **totals}), flush=True)
        time.sleep(0.05)

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "classifications.csv", index=False)
    sample[["debate_section_id", "question_date", "to_minister_or_department", "answer_status", "grouped_answer", "referred_or_direct_reply", "answer_length_band"]].to_csv(output_dir / "sample_manifest.csv", index=False)

    parsed = [json.loads(v) for v in out.get("classification_json", pd.Series(dtype=str)).tolist()]
    proposed = []
    tag_counts: dict[str, dict[str, int]] = {"question": {}, "answer": {}, "combined": {}}
    intent_counts: dict[str, int] = {}
    characteristic_counts: dict[str, int] = {}
    for section_row, result in zip(out.to_dict("records"), parsed):
        for q in result.get("questions", []):
            for tag in q.get("topic_tags", []): tag_counts["question"][tag] = tag_counts["question"].get(tag, 0) + 1
            for intent in q.get("question_intents", []): intent_counts[intent] = intent_counts.get(intent, 0) + 1
            for item in q.get("proposed_new_tags", []): proposed.append({"scope": "question", "debate_section_id": section_row["debate_section_id"], **item})
        for tag in result.get("answer", {}).get("topic_tags", []): tag_counts["answer"][tag] = tag_counts["answer"].get(tag, 0) + 1
        for char in result.get("answer", {}).get("answer_characteristics", []): characteristic_counts[char] = characteristic_counts.get(char, 0) + 1
        for item in result.get("answer", {}).get("proposed_new_tags", []): proposed.append({"scope": "answer", "debate_section_id": section_row["debate_section_id"], **item})
        for tag in result.get("combined_exchange", {}).get("topic_tags", []): tag_counts["combined"][tag] = tag_counts["combined"].get(tag, 0) + 1
        for item in result.get("combined_exchange", {}).get("proposed_new_tags", []): proposed.append({"scope": "combined", "debate_section_id": section_row["debate_section_id"], **item})

    proposed_df = pd.DataFrame(proposed)
    proposed_df.to_csv(output_dir / "proposed_new_tags.csv", index=False)
    summary = {
        "pilot_version": config["pilot_version"],
        "model": args.model,
        "target_sections": args.sample_size,
        "attempted_sections": int(len(sample)),
        "successful_sections": int(len(out)),
        "failed_sections": int(len(failures)),
        "validation_error_sections": int(out["validation_errors"].fillna("").ne("").sum()) if not out.empty else 0,
        "usage": totals,
        "average_tokens_per_successful_section": round(totals["total_tokens"] / len(out), 1) if len(out) else None,
        "question_topic_counts": dict(sorted(tag_counts["question"].items(), key=lambda x: (-x[1], x[0]))),
        "answer_topic_counts": dict(sorted(tag_counts["answer"].items(), key=lambda x: (-x[1], x[0]))),
        "combined_topic_counts": dict(sorted(tag_counts["combined"].items(), key=lambda x: (-x[1], x[0]))),
        "question_intent_counts": dict(sorted(intent_counts.items(), key=lambda x: (-x[1], x[0]))),
        "answer_characteristic_counts": dict(sorted(characteristic_counts.items(), key=lambda x: (-x[1], x[0]))),
        "proposed_new_tag_rows": int(len(proposed_df)),
        "failure_examples": failures[:20],
        "research_only": True,
        "production_changed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
