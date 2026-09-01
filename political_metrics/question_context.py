from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd


ORAL_QUESTION_SECTION_COLUMNS = [
    "debate_section_id",
    "debate_date",
    "section_heading",
    "oral_question_count",
    "question_ids_json",
    "asking_member_count",
    "related_speech_count",
    "related_speaker_count",
    "source_batch_id",
    "component_version",
    "calculated_at_utc",
    "contract_version",
]

SPEECH_CONTEXT_COLUMNS = [
    "speech_id",
    "debate_date",
    "debate_section_id",
    "speech_context",
    "is_oral_question_related",
    "oral_question_count_in_section",
    "related_question_ids_json",
    "source_batch_id",
    "component_version",
    "calculated_at_utc",
    "contract_version",
]


def _stamp(frame: pd.DataFrame, *, source_batch_id: str, contract_version: int) -> pd.DataFrame:
    result = frame.copy()
    result["source_batch_id"] = source_batch_id
    result["component_version"] = 1
    result["calculated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["contract_version"] = contract_version
    return result


def _oral_questions(questions: pd.DataFrame) -> pd.DataFrame:
    required = {"question_id", "question_type", "debate_section_id", "question_date"}
    missing = sorted(required - set(questions.columns))
    if missing:
        raise ValueError(f"questions missing required oral-question relationship columns: {missing}")
    data = questions.copy()
    qtype = data["question_type"].fillna("").astype(str).str.strip().str.lower()
    data = data[qtype.eq("oral")].copy()
    data = data[data["debate_section_id"].notna() & data["debate_section_id"].astype(str).str.strip().ne("")]
    return data


def build_oral_question_sections(
    *,
    questions: pd.DataFrame,
    speeches: pd.DataFrame,
    debate_sections: pd.DataFrame,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    oral = _oral_questions(questions)
    if oral.empty:
        return pd.DataFrame(columns=ORAL_QUESTION_SECTION_COLUMNS)

    oral["debate_section_id"] = oral["debate_section_id"].astype(str)
    oral["question_id"] = oral["question_id"].astype(str)
    oral["question_date"] = pd.to_datetime(oral["question_date"], errors="coerce")

    member_col = "asked_by_member_code" if "asked_by_member_code" in oral.columns else None
    records: list[dict] = []
    for section_id, group in oral.groupby("debate_section_id", sort=True):
        qids = sorted(group["question_id"].dropna().astype(str).unique().tolist())
        dates = pd.to_datetime(group["question_date"], errors="coerce").dropna()
        records.append(
            {
                "debate_section_id": str(section_id),
                "debate_date": dates.min().date().isoformat() if not dates.empty else None,
                "oral_question_count": len(qids),
                "question_ids_json": json.dumps(qids, separators=(",", ":")),
                "asking_member_count": int(group[member_col].dropna().astype(str).nunique()) if member_col else 0,
            }
        )
    result = pd.DataFrame(records)

    headings = pd.DataFrame(columns=["debate_section_id", "section_heading"])
    if "debate_section_id" in debate_sections.columns:
        heading_col = next((c for c in ["show_as", "heading", "title"] if c in debate_sections.columns), None)
        if heading_col:
            headings = debate_sections[["debate_section_id", heading_col]].copy()
            headings["debate_section_id"] = headings["debate_section_id"].astype(str)
            headings = headings.drop_duplicates("debate_section_id", keep="last").rename(columns={heading_col: "section_heading"})
    result = result.merge(headings, on="debate_section_id", how="left")
    result["section_heading"] = result.get("section_heading", "").fillna("")

    speech = speeches.copy()
    if {"speech_id", "debate_section_id"}.issubset(speech.columns):
        speech = speech[speech["debate_section_id"].notna()].copy()
        speech["debate_section_id"] = speech["debate_section_id"].astype(str)
        agg = speech.groupby("debate_section_id").agg(
            related_speech_count=("speech_id", "nunique"),
            related_speaker_count=("speaker_member_code", "nunique") if "speaker_member_code" in speech.columns else ("speech_id", "size"),
        ).reset_index()
        result = result.merge(agg, on="debate_section_id", how="left")
    else:
        result["related_speech_count"] = 0
        result["related_speaker_count"] = 0

    result["related_speech_count"] = pd.to_numeric(result.get("related_speech_count"), errors="coerce").fillna(0).astype(int)
    result["related_speaker_count"] = pd.to_numeric(result.get("related_speaker_count"), errors="coerce").fillna(0).astype(int)
    result = _stamp(result, source_batch_id=source_batch_id, contract_version=contract_version)
    return result[ORAL_QUESTION_SECTION_COLUMNS]


def build_speech_question_context(
    *,
    speeches: pd.DataFrame,
    questions: pd.DataFrame,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    required = {"speech_id", "debate_section_id", "debate_date"}
    missing = sorted(required - set(speeches.columns))
    if missing:
        raise ValueError(f"speeches missing required question-context columns: {missing}")

    oral = _oral_questions(questions)
    oral_map: dict[str, list[str]] = {}
    if not oral.empty:
        oral["debate_section_id"] = oral["debate_section_id"].astype(str)
        oral["question_id"] = oral["question_id"].astype(str)
        oral_map = (
            oral.groupby("debate_section_id")["question_id"]
            .apply(lambda values: sorted(set(values.astype(str))))
            .to_dict()
        )

    result = speeches[["speech_id", "debate_date", "debate_section_id"]].copy()
    result["speech_id"] = result["speech_id"].astype(str)
    result["debate_section_id"] = result["debate_section_id"].fillna("").astype(str)
    result["oral_question_count_in_section"] = result["debate_section_id"].map(lambda sid: len(oral_map.get(sid, [])))
    result["is_oral_question_related"] = result["oral_question_count_in_section"].gt(0)
    result["speech_context"] = result["is_oral_question_related"].map({True: "oral_question_exchange", False: "other"})
    result["related_question_ids_json"] = result["debate_section_id"].map(
        lambda sid: json.dumps(oral_map.get(sid, []), separators=(",", ":"))
    )
    result = _stamp(result, source_batch_id=source_batch_id, contract_version=contract_version)
    return result[SPEECH_CONTEXT_COLUMNS]
