#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any

import pandas as pd

import process.written_pq_semantic_pilot as pilot


def _strip_unsupported_json_schema_keywords(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_unsupported_json_schema_keywords(item)
            for key, item in value.items()
            if key != "uniqueItems"
        }
    if isinstance(value, list):
        return [_strip_unsupported_json_schema_keywords(item) for item in value]
    return value


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _quote_present(quote: str, source: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(source)


def leaf_topic_tags(config: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for topic in config["broad_topics"]:
        general = str(topic.get("general_tag") or "").strip()
        if general:
            tags.append(general)
        tags.extend(str(child) for child in topic.get("children", []))
    return tags


def topic_parent_map(config: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for topic in config["broad_topics"]:
        parent = str(topic["id"])
        general = str(topic.get("general_tag") or "").strip()
        if general:
            mapping[general] = parent
        for child in topic.get("children", []):
            mapping[str(child)] = parent
    return mapping


_original_schema = pilot.schema
_original_classify = pilot.classify
_original_build_exchange_frame = pilot.build_exchange_frame


def compatible_schema(config):
    value = _strip_unsupported_json_schema_keywords(_original_schema(config))
    value["properties"].pop("combined_exchange", None)
    value["required"] = [item for item in value["required"] if item != "combined_exchange"]
    return value


def clean_exchange_frame(questions, sections, bridge):
    frame = _original_build_exchange_frame(questions, sections, bridge)
    frame["answer_text"] = frame["answer_text"].map(_text)
    return frame


def strict_prompt(config: dict[str, Any], record: dict[str, Any]) -> list[dict[str, str]]:
    taxonomy = []
    for parent in config["broad_topics"]:
        taxonomy.append(f"CATEGORY {parent['id']}: {parent['label']} (category name is NOT a selectable tag)")
        general = str(parent.get("general_tag") or "").strip()
        if general:
            taxonomy.append(f"  - {general} [use only when this category is clearly relevant but no specific child tag fits]")
        taxonomy.extend(f"  - {child}" for child in parent.get("children", []))

    question_blocks = []
    for q in record["questions"]:
        question_blocks.append(
            "<QUESTION>\n"
            f"<QUESTION_ID>{q['question_id']}</QUESTION_ID>\n"
            f"<QUESTION_TEXT>{_text(q.get('question_text'))}</QUESTION_TEXT>\n"
            "</QUESTION>"
        )

    system = (
        "You are performing research-only semantic indexing of Irish Parliamentary Questions. "
        "Return ONLY per-QUESTION classifications and one ANSWER classification. The combined exchange view is derived later in code. "
        "Scope separation is mandatory. A QUESTION classification may use only that question's QUESTION_TEXT. "
        "The ANSWER classification may use only ANSWER_TEXT; never copy a topic, entity, or position from a question into the answer merely because the answer responds to it. "
        "Topic classification is multi-label, but choose only the selectable leaf/general tags listed beneath categories. Never output a CATEGORY id itself. "
        "Prefer a precise child tag over a *_general tag. Use a *_general tag only when the category is genuinely substantive and no child fits. "
        "Do not use government/public-service tags merely because a department, public body, agency, minister, referral, or reply exists; use them only when administration, governance, procurement, public-service staffing/delivery, state-agency structure, transparency, or local government is itself substantive. "
        "Referral/direct-reply mechanics belong in answer_characteristics and normally are not topic tags. "
        "transport_fares means fares, ticket prices, fare reductions, or public-transport charges. energy_prices means electricity, gas, heating, fuel, or other energy prices; never use energy_prices for transport fares. "
        "international_relations covers diplomacy and foreign-government engagement; defence tags require actual defence/security subject matter. "
        "equality_discrimination covers accessibility/equality issues without implying courts, policing, or criminal justice. "
        "Do not infer political support/opposition, truthfulness, quality, evasiveness, effectiveness, motives, or unstated positions. "
        "Known metadata such as department, TD, date and party are context only. Do not turn them into topics. "
        "Do not resolve pronouns or generic phrases such as 'my Department' into named entities from metadata; entity names must be explicitly present in the text for that scope. "
        "Only use proposed_new_tags when no approved tag accurately captures a useful recurring concept; never force a semantically wrong nearby tag. "
        "Every topic tag must have exactly matching topic_evidence with a short verbatim evidence quote from the SAME scope. "
        "Every entity/proposed-tag evidence_quote must also occur verbatim in that scope. "
        "If ANSWER_TEXT is empty or contains no substantive wording, return no answer topic tags/entities/evidence and use no_substantive_answer as appropriate."
    )
    repair_note = str(record.get("_repair_note") or "").strip()
    if repair_note:
        system += (
            " This is a repair attempt after deterministic validation failed with: " + repair_note + ". "
            "Correct the cited structural/evidence problems. In particular, copy evidence quotes exactly from the scoped source text and provide one evidence item for every topic tag."
        )

    user = (
        "SELECTABLE TOPIC TAXONOMY:\n" + "\n".join(taxonomy) +
        "\n\nQUESTION INTENTS:\n- " + "\n- ".join(config["question_intents"]) +
        "\n\nANSWER CHARACTERISTICS:\n- " + "\n- ".join(config["answer_characteristics"]) +
        "\n\nSOURCE METADATA (context only; do not recreate it as semantic output):\n" +
        str(record["metadata"]) +
        "\n\n" + "\n\n".join(question_blocks) +
        "\n\n<ANSWER>\n<ANSWER_TEXT>" + _text(record["answer_text"]) + "</ANSWER_TEXT>\n</ANSWER>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _normalize_questions(result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    expected = [str(q["question_id"]) for q in record["questions"]]
    first_by_id = {}
    for item in result.get("questions", []):
        qid = str(item.get("question_id", ""))
        if qid in expected and qid not in first_by_id:
            first_by_id[qid] = item
    result["questions"] = [first_by_id[qid] for qid in expected if qid in first_by_id]
    return result


def _derive_combined(result: dict[str, Any]) -> dict[str, Any]:
    ordered_tags: list[str] = []
    evidence_by_tag: dict[str, dict[str, str]] = {}
    proposed: list[dict[str, str]] = []
    seen_proposed: set[str] = set()

    scopes = list(result.get("questions", [])) + [result.get("answer", {})]
    for obj in scopes:
        for tag in obj.get("topic_tags", []):
            if tag not in ordered_tags:
                ordered_tags.append(tag)
        for item in obj.get("topic_evidence", []):
            tag = str(item.get("tag", ""))
            if tag and tag not in evidence_by_tag:
                evidence_by_tag[tag] = {"tag": tag, "evidence_quote": str(item.get("evidence_quote", ""))}
        for item in obj.get("proposed_new_tags", []):
            key = str(item.get("tag", "")).strip().casefold()
            if key and key not in seen_proposed:
                proposed.append(item)
                seen_proposed.add(key)

    result["combined_exchange"] = {
        "topic_tags": ordered_tags,
        "proposed_new_tags": proposed,
        "topic_evidence": [evidence_by_tag[tag] for tag in ordered_tags if tag in evidence_by_tag],
    }
    return result


def strict_validate_result(record: dict[str, Any], result: dict[str, Any], config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {str(q["question_id"]): _text(q.get("question_text")) for q in record["questions"]}
    actual_qids = [str(q.get("question_id", "")) for q in result.get("questions", [])]
    if sorted(expected) != sorted(actual_qids):
        errors.append("question_id_set_mismatch")

    allowed = set(leaf_topic_tags(config))

    def check_scope(scope: str, obj: dict[str, Any], source: str):
        tags = list(obj.get("topic_tags", []))
        if not set(tags).issubset(allowed):
            errors.append(f"invalid_{scope}_topic_tag")
        evidence = obj.get("topic_evidence", [])
        evidence_tags = [str(e.get("tag", "")) for e in evidence]
        if set(tags) != set(evidence_tags):
            errors.append(f"{scope}_topic_evidence_coverage_mismatch")
        for item in evidence:
            if not _quote_present(item.get("evidence_quote", ""), source):
                errors.append(f"{scope}_topic_evidence_not_in_scope")
                break
        for item in obj.get("entities", []):
            if not _quote_present(item.get("evidence_quote", ""), source):
                errors.append(f"{scope}_entity_evidence_not_in_scope")
                break
        for item in obj.get("proposed_new_tags", []):
            if not _quote_present(item.get("evidence_quote", ""), source):
                errors.append(f"{scope}_proposed_tag_evidence_not_in_scope")
                break

    for q in result.get("questions", []):
        qid = str(q.get("question_id", ""))
        check_scope("question", q, expected.get(qid, ""))

    answer_text = _text(record.get("answer_text"))
    check_scope("answer", result.get("answer", {}), answer_text)

    combined_source = " ".join(list(expected.values()) + [answer_text])
    check_scope("combined", result.get("combined_exchange", {}), combined_source)

    if not answer_text.strip():
        answer = result.get("answer", {})
        if answer.get("topic_tags") or answer.get("entities") or answer.get("topic_evidence") or answer.get("proposed_new_tags"):
            errors.append("empty_answer_has_semantic_output")

    return sorted(set(errors))


def _run_once(client, config, record, model):
    result, usage = _original_classify(client, config, record, model)
    result = _normalize_questions(result, record)
    result = _derive_combined(result)
    return result, usage


def normalized_classify(client, config, record, model):
    result, usage = _run_once(client, config, record, model)
    errors = strict_validate_result(record, result, config)
    repaired = False
    if errors:
        repair_record = dict(record)
        repair_record["_repair_note"] = ";".join(errors)
        repaired_result, repaired_usage = _run_once(client, config, repair_record, model)
        repaired_errors = strict_validate_result(record, repaired_result, config)
        usage = {key: int(usage.get(key, 0)) + int(repaired_usage.get(key, 0)) for key in usage}
        if len(repaired_errors) <= len(errors):
            result = repaired_result
            errors = repaired_errors
            repaired = True
    result["_pilot_meta"] = {"repair_attempted": repaired, "post_repair_validation_errors": errors}
    return result, usage


pilot.allowed_topic_tags = leaf_topic_tags
pilot.schema = compatible_schema
pilot.prompt = strict_prompt
pilot.classify = normalized_classify
pilot.build_exchange_frame = clean_exchange_frame
pilot.validate_result = strict_validate_result

if __name__ == "__main__":
    raise SystemExit(pilot.main())
