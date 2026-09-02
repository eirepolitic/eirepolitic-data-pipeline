from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pandas as pd


ORAL_QUESTION_SECTION_COLUMNS = [
    "debate_section_id",
    "debate_date",
    "section_heading",
    "oral_question_count",
    "question_ids_json",
    "asking_member_count",
    "participating_submitting_member_count",
    "participating_submitter_share",
    "ordinary_non_submitter_td_count",
    "grouped_exchange",
    "related_speech_count",
    "related_speaker_count",
    "related_speech_word_count",
    "ministerial_intervention_count",
    "ministerial_word_count",
    "ministerial_word_share",
    "chair_intervention_count",
    "chair_word_count",
    "ordinary_member_intervention_count",
    "ordinary_member_word_count",
    "collective_or_unidentified_intervention_count",
    "collective_or_unidentified_word_count",
    "source_batch_id",
    "component_version",
    "calculated_at_utc",
    "contract_version",
]

ORAL_QUESTION_EXCHANGE_PARTICIPANT_COLUMNS = [
    "debate_section_id",
    "debate_date",
    "participant_key",
    "member_code",
    "speaker_name",
    "participant_role",
    "is_recorded_submitter",
    "intervention_count",
    "word_count",
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

_CHAIR_RE = re.compile(
    r"ceann comhairle|leas-cheann comhairle|cathaoirleach|acting chair(?:man|person)?|chairman|chairperson",
    re.I,
)
_MINISTER_RE = re.compile(r"\bminister\b|\btaoiseach\b|\btánaiste\b|attorney general", re.I)


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


def _prepare_offices(offices: pd.DataFrame | None) -> pd.DataFrame:
    if offices is None or offices.empty:
        return pd.DataFrame(columns=["member_code", "office_name", "office_start", "office_end"])
    required = {"member_code", "office_name", "office_start", "office_end"}
    missing = sorted(required - set(offices.columns))
    if missing:
        raise ValueError(f"member offices missing required role columns: {missing}")
    result = offices[["member_code", "office_name", "office_start", "office_end"]].copy()
    result["member_code"] = result["member_code"].fillna("").astype(str)
    result["office_start"] = pd.to_datetime(result["office_start"], errors="coerce")
    result["office_end"] = pd.to_datetime(result["office_end"], errors="coerce")
    return result


def _participant_role(*, member_code: str, speaker_name: str, debate_date, offices: pd.DataFrame) -> str:
    name = str(speaker_name or "")
    if _CHAIR_RE.search(name):
        return "chair"
    if _MINISTER_RE.search(name):
        return "ministerial"
    code = str(member_code or "").strip()
    if not code:
        return "collective_or_unidentified"
    date = pd.to_datetime(debate_date, errors="coerce")
    active = offices[offices["member_code"].eq(code)]
    if not active.empty and pd.notna(date):
        mask = (active["office_start"].isna() | active["office_start"].le(date)) & (
            active["office_end"].isna() | active["office_end"].gt(date)
        )
        names = " ".join(active.loc[mask, "office_name"].dropna().astype(str))
        if _CHAIR_RE.search(names):
            return "chair"
        if _MINISTER_RE.search(names):
            return "ministerial"
    return "ordinary_member"


def _oral_exchange_speeches(
    *, speeches: pd.DataFrame, oral: pd.DataFrame, offices: pd.DataFrame | None
) -> pd.DataFrame:
    required = {"speech_id", "debate_section_id", "debate_date", "speaker_member_code", "speaker_name", "word_count"}
    missing = sorted(required - set(speeches.columns))
    if missing:
        raise ValueError(f"speeches missing required oral-exchange participant columns: {missing}")
    oral_sections = set(oral["debate_section_id"].astype(str))
    result = speeches[speeches["debate_section_id"].notna()].copy()
    result["debate_section_id"] = result["debate_section_id"].astype(str)
    result = result[result["debate_section_id"].isin(oral_sections)].copy()
    result["speech_id"] = result["speech_id"].astype(str)
    result["speaker_member_code"] = result["speaker_member_code"].fillna("").astype(str)
    result["speaker_name"] = result["speaker_name"].fillna("").astype(str)
    result["word_count_num"] = pd.to_numeric(result["word_count"], errors="coerce")
    if result["word_count_num"].isna().any():
        missing_count = int(result["word_count_num"].isna().sum())
        raise ValueError(f"oral-question exchange speeches contain {missing_count} missing/non-numeric word_count values")
    prepared_offices = _prepare_offices(offices)
    result["participant_role"] = [
        _participant_role(
            member_code=row.speaker_member_code,
            speaker_name=row.speaker_name,
            debate_date=row.debate_date,
            offices=prepared_offices,
        )
        for row in result.itertuples(index=False)
    ]
    return result


def build_oral_question_sections(
    *,
    questions: pd.DataFrame,
    speeches: pd.DataFrame,
    debate_sections: pd.DataFrame,
    member_offices: pd.DataFrame | None = None,
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
    submitter_map: dict[str, set[str]] = {}
    for section_id, group in oral.groupby("debate_section_id", sort=True):
        qids = sorted(group["question_id"].dropna().astype(str).unique().tolist())
        dates = pd.to_datetime(group["question_date"], errors="coerce").dropna()
        submitters = (
            set(x for x in group[member_col].dropna().astype(str) if str(x).strip()) if member_col else set()
        )
        submitter_map[str(section_id)] = submitters
        records.append(
            {
                "debate_section_id": str(section_id),
                "debate_date": dates.min().date().isoformat() if not dates.empty else None,
                "oral_question_count": len(qids),
                "question_ids_json": json.dumps(qids, separators=(",", ":")),
                "asking_member_count": len(submitters),
                "grouped_exchange": len(qids) > 1,
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

    speech = _oral_exchange_speeches(speeches=speeches, oral=oral, offices=member_offices)
    speech["is_recorded_submitter"] = [
        row.speaker_member_code in submitter_map.get(str(row.debate_section_id), set())
        for row in speech.itertuples(index=False)
    ]

    for section_id, group in speech.groupby("debate_section_id"):
        submitters = submitter_map.get(str(section_id), set())
        participating_submitters = set(
            group.loc[group["is_recorded_submitter"] & group["speaker_member_code"].ne(""), "speaker_member_code"].astype(str)
        )
        other_ordinary = set(
            group.loc[
                group["participant_role"].eq("ordinary_member")
                & ~group["is_recorded_submitter"]
                & group["speaker_member_code"].ne(""),
                "speaker_member_code",
            ].astype(str)
        )
        role_interventions = group.groupby("participant_role")["speech_id"].nunique().to_dict()
        role_words = group.groupby("participant_role")["word_count_num"].sum().to_dict()
        total_words = int(group["word_count_num"].sum())
        mask = result["debate_section_id"].eq(str(section_id))
        result.loc[mask, "participating_submitting_member_count"] = len(participating_submitters)
        result.loc[mask, "participating_submitter_share"] = (
            len(participating_submitters) / len(submitters) if submitters else None
        )
        result.loc[mask, "ordinary_non_submitter_td_count"] = len(other_ordinary)
        result.loc[mask, "related_speech_count"] = int(group["speech_id"].nunique())
        result.loc[mask, "related_speaker_count"] = int(
            group.loc[group["speaker_member_code"].ne(""), "speaker_member_code"].nunique()
        )
        result.loc[mask, "related_speech_word_count"] = total_words
        for role, prefix in [
            ("ministerial", "ministerial"),
            ("chair", "chair"),
            ("ordinary_member", "ordinary_member"),
            ("collective_or_unidentified", "collective_or_unidentified"),
        ]:
            result.loc[mask, f"{prefix}_intervention_count"] = int(role_interventions.get(role, 0))
            result.loc[mask, f"{prefix}_word_count"] = int(role_words.get(role, 0))
        result.loc[mask, "ministerial_word_share"] = (
            int(role_words.get("ministerial", 0)) / total_words if total_words else None
        )

    integer_cols = [
        "participating_submitting_member_count",
        "ordinary_non_submitter_td_count",
        "related_speech_count",
        "related_speaker_count",
        "related_speech_word_count",
        "ministerial_intervention_count",
        "ministerial_word_count",
        "chair_intervention_count",
        "chair_word_count",
        "ordinary_member_intervention_count",
        "ordinary_member_word_count",
        "collective_or_unidentified_intervention_count",
        "collective_or_unidentified_word_count",
    ]
    for col in integer_cols:
        result[col] = pd.to_numeric(result.get(col), errors="coerce").fillna(0).astype(int)
    result["participating_submitter_share"] = pd.to_numeric(result.get("participating_submitter_share"), errors="coerce")
    result["ministerial_word_share"] = pd.to_numeric(result.get("ministerial_word_share"), errors="coerce")
    result = _stamp(result, source_batch_id=source_batch_id, contract_version=contract_version)
    return result[ORAL_QUESTION_SECTION_COLUMNS]


def build_oral_question_exchange_participants(
    *,
    questions: pd.DataFrame,
    speeches: pd.DataFrame,
    member_offices: pd.DataFrame | None,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    oral = _oral_questions(questions)
    if oral.empty:
        return pd.DataFrame(columns=ORAL_QUESTION_EXCHANGE_PARTICIPANT_COLUMNS)
    oral["debate_section_id"] = oral["debate_section_id"].astype(str)
    submitter_map = {
        str(section_id): set(x for x in group["asked_by_member_code"].dropna().astype(str) if str(x).strip())
        if "asked_by_member_code" in group.columns else set()
        for section_id, group in oral.groupby("debate_section_id")
    }
    speech = _oral_exchange_speeches(speeches=speeches, oral=oral, offices=member_offices)
    speech["participant_key"] = speech["speaker_member_code"].where(
        speech["speaker_member_code"].ne(""), "name:" + speech["speaker_name"].astype(str)
    )
    speech["is_recorded_submitter"] = [
        row.speaker_member_code in submitter_map.get(str(row.debate_section_id), set())
        for row in speech.itertuples(index=False)
    ]

    records: list[dict] = []
    for (section_id, participant_key, participant_role), group in speech.groupby(
        ["debate_section_id", "participant_key", "participant_role"], sort=True, dropna=False
    ):
        dates = pd.to_datetime(group["debate_date"], errors="coerce").dropna()
        member_code = next((x for x in group["speaker_member_code"].astype(str) if x), "")
        speaker_name = next((x for x in group["speaker_name"].astype(str) if x), "")
        records.append(
            {
                "debate_section_id": str(section_id),
                "debate_date": dates.min().date().isoformat() if not dates.empty else None,
                "participant_key": str(participant_key),
                "member_code": member_code,
                "speaker_name": speaker_name,
                "participant_role": str(participant_role),
                "is_recorded_submitter": bool(group["is_recorded_submitter"].any()),
                "intervention_count": int(group["speech_id"].nunique()),
                "word_count": int(group["word_count_num"].sum()),
            }
        )
    result = pd.DataFrame(records)
    result = _stamp(result, source_batch_id=source_batch_id, contract_version=contract_version)
    return result[ORAL_QUESTION_EXCHANGE_PARTICIPANT_COLUMNS]


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
