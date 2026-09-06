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


_original_schema = pilot.schema
_original_classify = pilot.classify
_original_build_exchange_frame = pilot.build_exchange_frame


def compatible_schema(config):
    return _strip_unsupported_json_schema_keywords(_original_schema(config))


def clean_exchange_frame(questions, sections, bridge):
    frame = _original_build_exchange_frame(questions, sections, bridge)
    frame["answer_text"] = frame["answer_text"].map(_text)
    return frame


def strict_prompt(config: dict[str, Any], record: dict[str, Any]) -> list[dict[str, str]]:
    taxonomy = []
    for parent in config["broad_topics"]:
        taxonomy.append(f"{parent['id']}: {parent['label']}")
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
        "Return three strictly separated semantic views: each QUESTION, the ANSWER, and the COMBINED_EXCHANGE. "
        "Scope separation is mandatory. A QUESTION classification may use only that question's QUESTION_TEXT. "
        "The ANSWER classification may use only ANSWER_TEXT; never copy a topic, entity, or position from a question into the answer merely because the answer responds to it. "
        "COMBINED_EXCHANGE may use both questions and answer. "
        "Use multiple topic tags when genuinely relevant, preferring specific approved tags. "
        "Do not use government_public_service, state_agencies, or public_service_delivery merely because a department/public body exists, is named, responds, or receives a referral; use those tags only when public administration, agency governance, or service-delivery arrangements are themselves a substantive subject. "
        "Referral/direct-reply mechanics belong in answer_characteristics and normally are not topic tags. "
        "Do not infer political support/opposition, truthfulness, quality, evasiveness, effectiveness, motives, or unstated positions. "
        "Known metadata such as department, TD, date and party are supplied separately. Do not turn them into semantic topics. "
        "Do not resolve pronouns or generic phrases such as 'my Department' into named entities from metadata; entity names must be explicitly present in the text for that scope. "
        "Only use proposed_new_tags when the approved taxonomy genuinely lacks a useful recurring concept. "
        "Every topic tag must have a short verbatim evidence quote from the SAME scope, and every entity/proposed tag evidence_quote must also occur verbatim in that scope. "
        "If ANSWER_TEXT is empty or contains no substantive wording, return no answer topic tags/entities/evidence and use no_substantive_answer as appropriate."
    )
    user = (
        "APPROVED TOPIC TAXONOMY:\n" + "\n".join(taxonomy) +
        "\n\nQUESTION INTENTS:\n- " + "\n- ".join(config["question_intents"]) +
        "\n\nANSWER CHARACTERISTICS:\n- " + "\n- ".join(config["answer_characteristics"]) +
        "\n\nSOURCE METADATA (context only; do not recreate it as semantic output):\n" +
        str(record["metadata"]) +
        "\n\n" + "\n\n".join(question_blocks) +
        "\n\n<ANSWER>\n<ANSWER_TEXT>" + _text(record["answer_text"]) + "</ANSWER_TEXT>\n</ANSWER>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalized_classify(client, config, record, model):
    result, usage = _original_classify(client, config, record, model)
    expected = [str(q["question_id"]) for q in record["questions"]]
    first_by_id = {}
    for item in result.get("questions", []):
        qid = str(item.get("question_id", ""))
        if qid in expected and qid not in first_by_id:
            first_by_id[qid] = item
    result["questions"] = [first_by_id[qid] for qid in expected if qid in first_by_id]
    return result, usage


def strict_validate_result(record: dict[str, Any], result: dict[str, Any], config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {str(q["question_id"]): _text(q.get("question_text")) for q in record["questions"]}
    actual_qids = [str(q.get("question_id", "")) for q in result.get("questions", [])]
    if sorted(expected) != sorted(actual_qids):
        errors.append("question_id_set_mismatch")

    allowed = set(pilot.allowed_topic_tags(config))

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


pilot.schema = compatible_schema
pilot.prompt = strict_prompt
pilot.classify = normalized_classify
pilot.build_exchange_frame = clean_exchange_frame
pilot.validate_result = strict_validate_result

if __name__ == "__main__":
    raise SystemExit(pilot.main())
