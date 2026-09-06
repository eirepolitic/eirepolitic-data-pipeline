#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

import process.written_pq_semantic_pilot as pilot
import process.written_pq_semantic_pilot_compat as compat


def cheap_schema(config: dict[str, Any]) -> dict[str, Any]:
    tags = compat.leaf_topic_tags(config)
    question_item = {
        "type": "object",
        "properties": {
            "question_id": {"type": "string"},
            "topic_tags": {"type": "array", "items": {"type": "string", "enum": tags}},
            "question_intents": {"type": "array", "items": {"type": "string", "enum": config["question_intents"]}},
            "proposed_new_tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["question_id", "topic_tags", "question_intents", "proposed_new_tags"],
        "additionalProperties": False,
    }
    answer_obj = {
        "type": "object",
        "properties": {
            "topic_tags": {"type": "array", "items": {"type": "string", "enum": tags}},
            "answer_characteristics": {"type": "array", "items": {"type": "string", "enum": config["answer_characteristics"]}},
            "proposed_new_tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["topic_tags", "answer_characteristics", "proposed_new_tags"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "questions": {"type": "array", "items": question_item},
            "answer": answer_obj,
        },
        "required": ["questions", "answer"],
        "additionalProperties": False,
    }


def cheap_prompt(config: dict[str, Any], record: dict[str, Any]) -> list[dict[str, str]]:
    taxonomy = []
    for parent in config["broad_topics"]:
        taxonomy.append(f"CATEGORY {parent['id']}: {parent['label']} (not selectable)")
        general = str(parent.get("general_tag") or "").strip()
        if general:
            taxonomy.append(f"  - {general}")
        taxonomy.extend(f"  - {child}" for child in parent.get("children", []))

    qs = "\n\n".join(
        f"QUESTION_ID: {q['question_id']}\nQUESTION_TEXT: {compat._text(q.get('question_text'))}"
        for q in record["questions"]
    )
    system = (
        "Create a cheap semantic routing index for Irish Parliamentary Questions. "
        "Classify each QUESTION using only its own text and classify the ANSWER using only answer text. "
        "Return only routing fields: topic tags, question intents, answer characteristics, and optional proposed new tag names. "
        "Do not extract entities or evidence quotes in this pass. "
        "Choose selectable leaf/general tags only; never output CATEGORY ids. Prefer precise children over *_general. "
        "Do not copy a question topic into an answer unless the answer itself discusses it. "
        "Department/public-body presence alone is not a government/public-service topic. Referrals belong in answer characteristics. "
        "transport_fares is for public-transport fares; energy_prices is for electricity/gas/heating/fuel prices. "
        "international_relations is for diplomacy/foreign-government engagement. "
        "Use proposed_new_tags only when no approved tag accurately captures a useful recurring concept. "
        "Do not infer support/opposition, truthfulness, quality, evasiveness, effectiveness, motives, or unstated positions."
    )
    user = (
        "TOPIC TAXONOMY:\n" + "\n".join(taxonomy) +
        "\n\nQUESTION INTENTS:\n- " + "\n- ".join(config["question_intents"]) +
        "\n\nANSWER CHARACTERISTICS:\n- " + "\n- ".join(config["answer_characteristics"]) +
        "\n\n" + qs +
        "\n\nANSWER_TEXT: " + compat._text(record.get("answer_text"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def derive_combined(result: dict[str, Any]) -> dict[str, Any]:
    tags: list[str] = []
    proposed: list[str] = []
    for obj in list(result.get("questions", [])) + [result.get("answer", {})]:
        for tag in obj.get("topic_tags", []):
            if tag not in tags:
                tags.append(tag)
        for tag in obj.get("proposed_new_tags", []):
            if tag and tag not in proposed:
                proposed.append(tag)
    result["combined_exchange"] = {"topic_tags": tags, "proposed_new_tags": proposed, "topic_evidence": []}
    return result


def classify(client, config, record, model):
    response = client.responses.create(
        model=model,
        input=cheap_prompt(config, record),
        reasoning={"effort": "low"},
        text={"verbosity": "low", "format": {"type": "json_schema", "name": "written_pq_semantic_route", "strict": True, "schema": cheap_schema(config)}},
        max_output_tokens=1600,
        store=False,
    )
    result = json.loads(str(response.output_text or "").strip())
    result = compat._normalize_questions(result, record)
    result = derive_combined(result)
    usage = getattr(response, "usage", None)
    return result, {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def validate(record: dict[str, Any], result: dict[str, Any], config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = [str(q["question_id"]) for q in record["questions"]]
    actual = [str(q.get("question_id", "")) for q in result.get("questions", [])]
    if sorted(expected) != sorted(actual):
        errors.append("question_id_set_mismatch")
    allowed = set(compat.leaf_topic_tags(config))
    for q in result.get("questions", []):
        if not set(q.get("topic_tags", [])).issubset(allowed):
            errors.append("invalid_question_topic_tag")
    if not set(result.get("answer", {}).get("topic_tags", [])).issubset(allowed):
        errors.append("invalid_answer_topic_tag")
    if not compat._text(record.get("answer_text")).strip() and result.get("answer", {}).get("topic_tags"):
        errors.append("empty_answer_has_topic_tags")
    return sorted(set(errors))


pilot.schema = cheap_schema
pilot.prompt = cheap_prompt
pilot.classify = classify
pilot.validate_result = validate
pilot.build_exchange_frame = compat.clean_exchange_frame
pilot.allowed_topic_tags = compat.leaf_topic_tags

if __name__ == "__main__":
    raise SystemExit(pilot.main())
