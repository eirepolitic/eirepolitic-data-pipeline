"""Helpers for parsing Oireachtas Akoma Ntoso debate XML."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterable

from .normalize import stable_hash

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


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
    speech_text: str
    language: str | None


def parse_debate_xml(*, xml_bytes: bytes, debate_id: str, debate_date: str | None) -> tuple[list[ParsedSpeech], dict[str, Any]]:
    """Parse debate XML into ordered speech records and parser diagnostics."""
    root = ET.fromstring(xml_bytes)
    diagnostics: dict[str, Any] = {
        "root_tag": _local(root.tag),
        "tag_counts": _tag_counts(root),
        "speech_tag_candidates": [],
        "section_tag_candidates": [],
    }

    rows: list[ParsedSpeech] = []
    context = _WalkContext(debate_id=debate_id, debate_date=debate_date)
    _walk(root, context=context, rows=rows, diagnostics=diagnostics)
    return rows, diagnostics


class _WalkContext:
    def __init__(self, *, debate_id: str, debate_date: str | None) -> None:
        self.debate_id = debate_id
        self.debate_date = debate_date
        self.section_stack: list[tuple[str | None, str | None]] = []
        self.speech_order = 0
        self.doc_lang: str | None = None

    @property
    def current_section(self) -> tuple[str | None, str | None]:
        return self.section_stack[-1] if self.section_stack else (None, None)


def _walk(element: ET.Element, *, context: _WalkContext, rows: list[ParsedSpeech], diagnostics: dict[str, Any]) -> None:
    local = _local(element.tag)
    if not context.doc_lang:
        context.doc_lang = element.attrib.get(XML_LANG) or element.attrib.get("lang")

    pushed_section = False
    if _is_section(local):
        section_eid = _first_attr(element, "eId", "eid", "id")
        section_id = _section_id(context.debate_id, section_eid)
        context.section_stack.append((section_id, section_eid))
        pushed_section = True
        if len(diagnostics["section_tag_candidates"]) < 20:
            diagnostics["section_tag_candidates"].append({"tag": local, "attrs": dict(element.attrib)})

    if _is_speech(local):
        text = _normalise_text(" ".join(element.itertext()))
        if text:
            context.speech_order += 1
            section_id, section_eid = context.current_section
            speaker_ref = _speaker_ref(element)
            speaker_name = _speaker_name(element, speaker_ref=speaker_ref)
            language = element.attrib.get(XML_LANG) or element.attrib.get("lang") or context.doc_lang
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
                    speech_text=text,
                    language=language,
                )
            )
            if len(diagnostics["speech_tag_candidates"]) < 20:
                diagnostics["speech_tag_candidates"].append({"tag": local, "attrs": dict(element.attrib), "text_prefix": text[:180]})
        return

    for child in list(element):
        _walk(child, context=context, rows=rows, diagnostics=diagnostics)

    if pushed_section:
        context.section_stack.pop()


def _is_section(local: str) -> bool:
    return local in {"debateSection", "section", "subsection"}


def _is_speech(local: str) -> bool:
    return local in {"speech"}


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
    if speaker_ref:
        text = speaker_ref.strip().lstrip("#")
        text = re.sub(r"[_-]+", " ", text)
        return text or None
    return None


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
