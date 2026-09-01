import json

import pandas as pd

from political_metrics.question_context import build_oral_question_sections, build_speech_question_context


def _questions():
    return pd.DataFrame(
        [
            {
                "question_id": "q1",
                "question_type": "Oral",
                "debate_section_id": "sec-a",
                "question_date": "2026-07-01",
                "asked_by_member_code": "m1",
            },
            {
                "question_id": "q2",
                "question_type": "Oral",
                "debate_section_id": "sec-a",
                "question_date": "2026-07-01",
                "asked_by_member_code": "m2",
            },
            {
                "question_id": "q3",
                "question_type": "Written",
                "debate_section_id": "sec-written",
                "question_date": "2026-07-01",
                "asked_by_member_code": "m1",
            },
        ]
    )


def _speeches():
    return pd.DataFrame(
        [
            {"speech_id": "s1", "debate_date": "2026-07-01", "debate_section_id": "sec-a", "speaker_member_code": "m1"},
            {"speech_id": "s2", "debate_date": "2026-07-01", "debate_section_id": "sec-a", "speaker_member_code": "minister"},
            {"speech_id": "s3", "debate_date": "2026-07-01", "debate_section_id": "sec-a", "speaker_member_code": "m2"},
            {"speech_id": "s4", "debate_date": "2026-07-01", "debate_section_id": "sec-other", "speaker_member_code": "m3"},
            {"speech_id": "s5", "debate_date": "2026-07-01", "debate_section_id": "sec-written", "speaker_member_code": "m4"},
        ]
    )


def test_oral_question_section_counts_questions_and_speeches_once():
    sections = pd.DataFrame([{"debate_section_id": "sec-a", "show_as": "Urban Development"}])
    result = build_oral_question_sections(
        questions=_questions(),
        speeches=_speeches(),
        debate_sections=sections,
        source_batch_id="batch-1",
        contract_version=1,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["debate_section_id"] == "sec-a"
    assert row["section_heading"] == "Urban Development"
    assert row["oral_question_count"] == 2
    assert row["related_speech_count"] == 3
    assert row["related_speaker_count"] == 3
    assert row["asking_member_count"] == 2
    assert json.loads(row["question_ids_json"]) == ["q1", "q2"]


def test_written_question_does_not_create_oral_question_section():
    result = build_oral_question_sections(
        questions=_questions(),
        speeches=_speeches(),
        debate_sections=pd.DataFrame(columns=["debate_section_id", "show_as"]),
        source_batch_id="batch-1",
        contract_version=1,
    )

    assert set(result["debate_section_id"]) == {"sec-a"}
    assert "sec-written" not in set(result["debate_section_id"])


def test_speech_context_marks_only_speeches_in_oral_question_sections():
    result = build_speech_question_context(
        speeches=_speeches(),
        questions=_questions(),
        source_batch_id="batch-1",
        contract_version=1,
    ).set_index("speech_id")

    assert len(result) == 5
    for speech_id in ["s1", "s2", "s3"]:
        assert bool(result.loc[speech_id, "is_oral_question_related"]) is True
        assert result.loc[speech_id, "speech_context"] == "oral_question_exchange"
        assert result.loc[speech_id, "oral_question_count_in_section"] == 2
        assert json.loads(result.loc[speech_id, "related_question_ids_json"]) == ["q1", "q2"]

    for speech_id in ["s4", "s5"]:
        assert bool(result.loc[speech_id, "is_oral_question_related"]) is False
        assert result.loc[speech_id, "speech_context"] == "other"
        assert result.loc[speech_id, "oral_question_count_in_section"] == 0
        assert json.loads(result.loc[speech_id, "related_question_ids_json"]) == []


def test_speech_context_has_one_row_per_speech():
    result = build_speech_question_context(
        speeches=_speeches(),
        questions=_questions(),
        source_batch_id="batch-1",
        contract_version=1,
    )
    assert result["speech_id"].is_unique
    assert set(result["speech_id"]) == set(_speeches()["speech_id"])
