"""Helpers for parsing Oireachtas Akoma Ntoso debate XML."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from .normalize import stable_hash

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
NON_JOIN_SECTION_NAMES = {"prelude", "division", "ta", "nil", "staon"}


@dataclass(frozen=True)
class ParsedSpeech:
    speech_id: str
    debate_id: str
    debate_section_id: str | None
    section_eid: str | None
    debate_date: str | None
    speech_order: int
    speaker_ref: str | None
    speaker_name: str | None
    speaker_member_code: str | None
    speech_text: str
    language: str | None


def parse_debate_xml(
    *,
    xml_bytes: bytes,
    debate_id: str,
    debate_date: str | None,
    default_language: str | None = "en",
) -> tuple[list[ParsedSpeech], dict[str, Any]]:
    """Parse debate XML into ordered speech records and parser diagnostics."""
    root = ET.fromstring(xml_bytes)
    people = _person_references(root)
    diagnostics: dict[str, Any] = {
        "root_tag": _local(root.tag),
        "tag_counts": _tag_counts(root),
        "speech_tag_candidates": [],
        "section_tag_candidates": [],
        "person_reference_samples": list(people.values())[:20],
        "person_reference_count": len(people),
    }

    rows: list[ParsedSpeech] = []
    context = _WalkContext(
        debate_id=debate_id,
        debate_date=debate_date,
        default_language=default_language,
        people=people,
    )
    _walk(root, context=context, rows=rows, diagnostics=diagnostics)
    return rows, diagnostics


class _WalkContext:
    def __init__(
        self,
        *,
        debate_id: str,
        debate_date: str | None,
        default_language: str | None,
        people: dict[str, dict[str, str | None]],
    ) -> None:
        self.debate_id = debate_id
        self.debate_date = debate_date
        self.default_language = default_language
        self.people = people
        self.section_stack: list[tuple[str | None, str | None, str | None]] = []
        self.speech_order = 0
        self.doc_lang: str | None = None

    @property
    def join_section(self) -> tuple[str | None, str | None]:
        for section_id, section_eid, section_name in reversed(self.section_stack):
            if (section_name or "").lower() not in NON_JOIN_SECTION_NAMES:
                return section_id, section_eid
        if self.section_stack:
            section_id, section_eid, _ = self.section_stack[0]
            return section_id, section_eid
        return None, None


def _walk(element: ET.Element, *, context: _WalkContext, rows: list[ParsedSpeech], diagnostics: dict[str, Any]) -> None:
    local = _local(element.tag)
    if not context.doc_lang:
        context.doc_lang = element.attrib.get(XML_LANG) or element.attrib.get("lang")

    pushed_section = False
    if _is_section(local):
        section_eid = _first_attr(element, "eId", "eid", "id")
        section_name = _first_attr(element, "name")
        section_id = _section_id(context.debate_id, section_eid)
        context.section_stack.append((section_id, section_eid, section_name))
        pushed_section = True
        if len(diagnostics["section_tag_candidates"]) < 30:
            diagnostics["section_tag_candidates"].append({"tag": local, "attrs": dict(element.attrib)})

    if _is_speech(local):
        text = _speech_text(element)
        if text:
            context.speech_order += 1
            section_id, section_eid = context.join_section
            speaker_ref = _speaker_ref(element)
            person = context.people.get((speaker_ref or "").lstrip("#"), {})
            speaker_name = _speaker_name(element, speaker_ref=speaker_ref) or person.get("show_as")
            speaker_member_code = person.get("member_code")
            language = (
                element.attrib.get(XML_LANG)
                or element.attrib.get("lang")
                or context.doc_lang
                or context.default_language
            )
            speech_id = f"speech:{stable_hash([context.debate_id, section_id, context.speech_order, speaker_ref, text], length=24)}"
            rows.append(
                ParsedSpeech(
                    speech_id=speech_id,
                    debate_id=context.debate_id,
                    debate_section_id=section_id,
                    section_eid=section_eid,
                    debate_date=context.debate_date,
                    speech_order=context.speech_order,
                    speaker_ref=speaker_ref,
                    speaker_name=speaker_name,
                    speaker_member_code=speaker_member_code,
                    speech_text=text,
                    language=language,
                )
            )
            if len(diagnostics["speech_tag_candidates"]) < 20:
                diagnostics["speech_tag_candidates"].append(
                    {
                        "tag": local,
                        "attrs": dict(element.attrib),
                        "speaker_member_code": speaker_member_code,
                        "text_prefix": text[:180],
                    }
                )
        return

    for child in list(element):
        _walk(child, context=context, rows=rows, diagnostics=diagnostics)

    if pushed_section:
        context.section_stack.pop()


def _person_references(root: ET.Element) -> dict[str, dict[str, str | None]]:
    people: dict[str, dict[str, str | None]] = {}
    for element in root.iter():
        if _local(element.tag) != "TLCPerson":
            continue
        eid = _first_attr(element, "eId", "eid", "id")
        if not eid:
            continue
        href = _first_attr(element, "href", "refersTo", "uri")
        show_as = _first_attr(element, "showAs", "name")
        people[eid.lstrip("#")] = {
            "e_id": eid,
            "href": href,
            "show_as": show_as,
            "member_code": _member_code_from_href(href),
        }
    return people


def _member_code_from_href(href: str | None) -> str | None:
    if not href:
        return None
    marker = "/member/id/"
    if marker not in href:
        return None
    candidate = href.split(marker, 1)[1].split("/", 1)[0].strip()
    return candidate or None


def _is_section(local: str) -> bool:
    return local in {"debateSection", "section", "subsection"}


def _is_speech(local: str) -> bool:
    return local == "speech"


def _speaker_ref(element: ET.Element) -> str | None:
    value = _first_attr(element, "by", "as", "refersTo", "source")
    if value:
        return value
    for child in list(element):
        if _local(child.tag) in {"from", "docProponent", "speaker"}:
            ref = _first_attr(child, "href", "refersTo", "by", "as")
            if ref:
                return ref
    return None


def _speaker_name(element: ET.Element, *, speaker_ref: str | None) -> str | None:
    for child in list(element):
        if _local(child.tag) in {"from", "docProponent", "speaker"}:
            text = _normalise_text(" ".join(child.itertext()))
            if text:
                return text
    if speaker_ref and speaker_ref != "#":
        text = speaker_ref.strip().lstrip("#")
        text = re.sub(r"[_-]+", " ", text)
        return text or None
    return None


def _speech_text(element: ET.Element) -> str:
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in list(element):
        local = _local(child.tag)
        if local not in {"from", "recordedTime"}:
            parts.extend(child.itertext())
        if child.tail:
            parts.append(child.tail)
    return _normalise_text(" ".join(parts))


def _section_id(debate_id: str, section_eid: str | None) -> str | None:
    if not section_eid:
        return None
    base = debate_id.rsplit("/", 1)[0] if "/" in debate_id else debate_id
    return f"{base}/{section_eid}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _first_attr(element: ET.Element, *keys: str) -> str | None:
    for key in keys:
        value = element.attrib.get(key)
        if value:
            return value.strip()
    return None


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tag_counts(root: ET.Element) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in root.iter():
        local = _local(element.tag)
        counts[local] = counts.get(local, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:50])
