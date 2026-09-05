from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from lxml import etree


ANSWER_SECTION_COLUMNS = [
    "debate_section_id",
    "answer_date",
    "section_heading",
    "answer_status",
    "answer_text",
    "respondent_ref",
    "respondent_role_ref",
    "observed_question_count",
    "observed_question_eids_json",
    "grouped_answer",
    "referred_or_direct_reply",
    "summary_texts_json",
    "embedded_table_count",
    "source_xml_url",
    "source_xml_uri",
    "source_batch_id",
    "answer_version",
    "calculated_at_utc",
    "contract_version",
]

QUESTION_BRIDGE_COLUMNS = [
    "question_id",
    "debate_section_id",
    "question_date",
    "question_xml_match_status",
    "observed_question_eid",
    "source_xml_url",
    "source_batch_id",
    "bridge_version",
    "calculated_at_utc",
    "contract_version",
]

ANSWER_VERSION = 1
BRIDGE_VERSION = 1


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if isinstance(tag, str) and "}" in tag else str(tag)


def _clean_text(element) -> str:
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def _eid(element) -> str:
    for key, value in element.attrib.items():
        if _local(key) == "eId":
            return str(value)
    return ""


def _attr(element, name: str) -> str:
    for key, value in element.attrib.items():
        if _local(key) == name:
            return str(value)
    return ""


def _suffix(value: str) -> str:
    return str(value or "").rstrip("/").split("/")[-1]


def _json(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _referral_flag(text: str) -> bool:
    t = text.lower()
    patterns = [
        "for direct reply",
        "asked the health service executive to respond",
        "asked the hse to respond",
        "asked the national transport authority to respond",
        "asked irish rail to respond",
        "asked the relevant body to respond",
        "referred your question",
        "referred the question",
        "reply directly to the deputy",
        "respond directly to the deputy",
        "directly to the deputy",
    ]
    return any(p in t for p in patterns)


def _reply_not_received(text: str) -> bool:
    return "reply not received from department" in text.lower()


@dataclass(frozen=True)
class ParsedWrittenAnswer:
    debate_section_eid: str
    section_heading: str
    answer_status: str
    answer_text: str
    respondent_ref: str
    respondent_role_ref: str
    observed_question_eids: tuple[str, ...]
    grouped_answer: bool
    referred_or_direct_reply: bool
    summaries: tuple[str, ...]
    embedded_table_count: int


def parse_written_answer_xml(xml_bytes: bytes) -> ParsedWrittenAnswer:
    root = etree.fromstring(xml_bytes)
    sections = [el for el in root.iter() if _local(el.tag) == "debateSection"]
    if len(sections) != 1:
        raise ValueError(f"expected one debateSection, found {len(sections)}")
    section = sections[0]
    if _attr(section, "name") != "writtenAnswer":
        raise ValueError(f"expected writtenAnswer section, found {_attr(section, 'name')!r}")

    headings = [_clean_text(el) for el in section.iter() if _local(el.tag) == "heading"]
    qels = [el for el in section.iter() if _local(el.tag) == "question"]
    speeches = [el for el in section.iter() if _local(el.tag) == "speech"]
    summaries = tuple(_clean_text(el) for el in section.iter() if _local(el.tag) == "summary" and _clean_text(el))
    full_text = _clean_text(section)

    if speeches:
        answer_status = "ministerial_reply_present"
        answer_text = " ".join(_clean_text(el) for el in speeches if _clean_text(el)).strip()
        respondent_ref = _attr(speeches[0], "by")
        respondent_role_ref = _attr(speeches[0], "as")
    elif _reply_not_received(full_text):
        answer_status = "reply_not_received"
        answer_text = ""
        respondent_ref = ""
        respondent_role_ref = ""
    else:
        answer_status = "unresolved_structure"
        answer_text = ""
        respondent_ref = ""
        respondent_role_ref = ""

    question_eids = tuple(e for e in (_eid(el) for el in qels) if e)
    grouped = len(qels) > 1 or any("answered with question" in s.lower() or "answered with" in s.lower() for s in summaries)
    return ParsedWrittenAnswer(
        debate_section_eid=_eid(section),
        section_heading=headings[0] if headings else "",
        answer_status=answer_status,
        answer_text=answer_text,
        respondent_ref=respondent_ref,
        respondent_role_ref=respondent_role_ref,
        observed_question_eids=question_eids,
        grouped_answer=grouped,
        referred_or_direct_reply=_referral_flag(full_text),
        summaries=summaries,
        embedded_table_count=sum(1 for el in section.iter() if _local(el.tag) == "table"),
    )


def build_written_answer_foundations(
    *,
    written_questions: pd.DataFrame,
    xml_by_url: dict[str, bytes],
    source_batch_id: str,
    contract_version: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    now = datetime.now(timezone.utc).isoformat()
    questions = written_questions.copy()
    questions = questions[questions["question_type"].fillna("").astype(str).str.strip().str.lower().eq("written")].copy()
    if questions.empty:
        return (
            pd.DataFrame(columns=ANSWER_SECTION_COLUMNS),
            pd.DataFrame(columns=QUESTION_BRIDGE_COLUMNS),
            {"ready": True, "written_question_rows": 0, "section_rows": 0, "bridge_rows": 0},
        )

    required = {"question_id", "question_date", "debate_section_id", "source_xml_url"}
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(f"written questions missing required columns: {sorted(missing)}")

    section_rows: list[dict] = []
    bridge_rows: list[dict] = []
    parse_failures: list[dict] = []

    grouped = questions.groupby(["debate_section_id", "source_xml_url"], dropna=False, sort=False)
    for (section_id, url), group in grouped:
        section_id = str(section_id or "")
        url = str(url or "")
        if not section_id or not url:
            parse_failures.append({"debate_section_id": section_id, "source_xml_url": url, "error": "missing_section_or_url"})
            continue
        xml = xml_by_url.get(url)
        if not xml:
            parse_failures.append({"debate_section_id": section_id, "source_xml_url": url, "error": "xml_not_available"})
            continue
        try:
            parsed = parse_written_answer_xml(xml)
        except Exception as exc:
            parse_failures.append({"debate_section_id": section_id, "source_xml_url": url, "error": f"{type(exc).__name__}: {exc}"})
            continue

        expected_section_eid = _suffix(section_id)
        if parsed.debate_section_eid != expected_section_eid:
            parse_failures.append({
                "debate_section_id": section_id,
                "source_xml_url": url,
                "error": f"section_eid_mismatch:{parsed.debate_section_eid}:{expected_section_eid}",
            })
            continue

        answer_date = str(group["question_date"].dropna().astype(str).min() if group["question_date"].notna().any() else "")
        source_xml_uri = ""
        if "source_xml_uri" in group.columns:
            values = [str(v) for v in group["source_xml_uri"].dropna().tolist() if str(v)]
            source_xml_uri = values[0] if values else ""
        observed = tuple(parsed.observed_question_eids)
        section_rows.append({
            "debate_section_id": section_id,
            "answer_date": answer_date,
            "section_heading": parsed.section_heading,
            "answer_status": parsed.answer_status,
            "answer_text": parsed.answer_text,
            "respondent_ref": parsed.respondent_ref,
            "respondent_role_ref": parsed.respondent_role_ref,
            "observed_question_count": len(observed),
            "observed_question_eids_json": _json(observed),
            "grouped_answer": bool(parsed.grouped_answer),
            "referred_or_direct_reply": bool(parsed.referred_or_direct_reply),
            "summary_texts_json": _json(parsed.summaries),
            "embedded_table_count": int(parsed.embedded_table_count),
            "source_xml_url": url,
            "source_xml_uri": source_xml_uri,
            "source_batch_id": source_batch_id,
            "answer_version": ANSWER_VERSION,
            "calculated_at_utc": now,
            "contract_version": contract_version,
        })

        observed_set = set(observed)
        for row in group.to_dict("records"):
            qid = str(row.get("question_id") or "")
            q_eid = _suffix(qid)
            match_status = "question_id_matched_in_xml" if q_eid in observed_set else "section_matched_question_id_unmatched"
            bridge_rows.append({
                "question_id": qid,
                "debate_section_id": section_id,
                "question_date": str(row.get("question_date") or ""),
                "question_xml_match_status": match_status,
                "observed_question_eid": q_eid if q_eid in observed_set else "",
                "source_xml_url": url,
                "source_batch_id": source_batch_id,
                "bridge_version": BRIDGE_VERSION,
                "calculated_at_utc": now,
                "contract_version": contract_version,
            })

    sections = pd.DataFrame(section_rows, columns=ANSWER_SECTION_COLUMNS)
    bridge = pd.DataFrame(bridge_rows, columns=QUESTION_BRIDGE_COLUMNS)
    audit = audit_written_answer_foundations(
        written_questions=questions,
        answer_sections=sections,
        question_bridge=bridge,
        parse_failures=parse_failures,
    )
    return sections, bridge, audit


def audit_written_answer_foundations(
    *,
    written_questions: pd.DataFrame,
    answer_sections: pd.DataFrame,
    question_bridge: pd.DataFrame,
    parse_failures: list[dict] | None = None,
) -> dict:
    parse_failures = parse_failures or []
    section_dupes = int(answer_sections.duplicated(["debate_section_id"]).sum()) if not answer_sections.empty else 0
    bridge_dupes = int(question_bridge.duplicated(["question_id"]).sum()) if not question_bridge.empty else 0
    section_ids = set(answer_sections["debate_section_id"].astype(str)) if not answer_sections.empty else set()
    orphan_bridge = int((~question_bridge["debate_section_id"].astype(str).isin(section_ids)).sum()) if not question_bridge.empty else 0
    allowed_status = {"ministerial_reply_present", "reply_not_received", "unresolved_structure"}
    invalid_status = int((~answer_sections["answer_status"].isin(allowed_status)).sum()) if not answer_sections.empty else 0
    allowed_match = {"question_id_matched_in_xml", "section_matched_question_id_unmatched"}
    invalid_match = int((~question_bridge["question_xml_match_status"].isin(allowed_match)).sum()) if not question_bridge.empty else 0
    source_written = int(len(written_questions))
    bridge_count = int(len(question_bridge))
    missing_bridge = source_written - bridge_count
    answer_present = int(answer_sections["answer_status"].eq("ministerial_reply_present").sum()) if not answer_sections.empty else 0
    reply_missing = int(answer_sections["answer_status"].eq("reply_not_received").sum()) if not answer_sections.empty else 0
    unresolved = int(answer_sections["answer_status"].eq("unresolved_structure").sum()) if not answer_sections.empty else 0
    unmatched_question_ids = int(question_bridge["question_xml_match_status"].eq("section_matched_question_id_unmatched").sum()) if not question_bridge.empty else 0
    checks = {
        "section_primary_key_unique": section_dupes == 0,
        "bridge_primary_key_unique": bridge_dupes == 0,
        "bridge_sections_exist": orphan_bridge == 0,
        "answer_status_valid": invalid_status == 0,
        "question_match_status_valid": invalid_match == 0,
        "all_written_questions_bridged": missing_bridge == 0,
        "no_parse_failures": len(parse_failures) == 0,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "written_question_rows": source_written,
        "section_rows": int(len(answer_sections)),
        "bridge_rows": bridge_count,
        "ministerial_reply_sections": answer_present,
        "reply_not_received_sections": reply_missing,
        "unresolved_structure_sections": unresolved,
        "grouped_answer_sections": int(answer_sections["grouped_answer"].fillna(False).astype(bool).sum()) if not answer_sections.empty else 0,
        "referred_or_direct_reply_sections": int(answer_sections["referred_or_direct_reply"].fillna(False).astype(bool).sum()) if not answer_sections.empty else 0,
        "sections_with_embedded_tables": int((pd.to_numeric(answer_sections["embedded_table_count"], errors="coerce").fillna(0) > 0).sum()) if not answer_sections.empty else 0,
        "question_id_unmatched_bridge_rows": unmatched_question_ids,
        "parse_failure_count": len(parse_failures),
        "parse_failure_examples": parse_failures[:20],
        "section_duplicate_rows": section_dupes,
        "bridge_duplicate_rows": bridge_dupes,
        "orphan_bridge_rows": orphan_bridge,
        "missing_bridge_rows": missing_bridge,
    }
