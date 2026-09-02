import json

import pandas as pd

from political_metrics.question_context import (
    build_oral_question_exchange_participants,
    build_oral_question_sections,
    build_speech_question_context,
)


def _questions():
    return pd.DataFrame(
        [
            {"question_id": "q1", "question_type": "Oral", "debate_section_id": "sec-a", "question_date": "2026-07-01", "asked_by_member_code": "m1"},
            {"question_id": "q2", "question_type": "Oral", "debate_section_id": "sec-a", "question_date": "2026-07-01", "asked_by_member_code": "m2"},
            {"question_id": "q3", "question_type": "Written", "debate_section_id": "sec-written", "question_date": "2026-07-01", "asked_by_member_code": "m1"},
        ]
    )


def _speeches():
    return pd.DataFrame(
        [
            {"speech_id": "s1", "debate_date": "2026-07-01", "debate_section_id": "sec-a", "speaker_member_code": "m1", "speaker_name": "Deputy One", "word_count": 10},
            {"speech_id": "s2", "debate_date": "2026-07-01", "debate_section_id": "sec-a", "speaker_member_code": "minister", "speaker_name": "Minister for Testing", "word_count": 30},
            {"speech_id": "s3", "debate_date": "2026-07-01", "debate_section_id": "sec-a", "speaker_member_code": "m3", "speaker_name": "Deputy Three", "word_count": 20},
            {"speech_id": "s4", "debate_date": "2026-07-01", "debate_section_id": "sec-a", "speaker_member_code": "chair1", "speaker_name": "Acting Chairman", "word_count": 2},
            {"speech_id": "s5", "debate_date": "2026-07-01", "debate_section_id": "sec-a", "speaker_member_code": "", "speaker_name": "Deputies", "word_count": 2},
            {"speech_id": "s6", "debate_date": "2026-07-01", "debate_section_id": "sec-other", "speaker_member_code": "m3", "speaker_name": "Deputy Three", "word_count": 5},
            {"speech_id": "s7", "debate_date": "2026-07-01", "debate_section_id": "sec-written", "speaker_member_code": "m4", "speaker_name": "Deputy Four", "word_count": 5},
        ]
    )


def _offices():
    return pd.DataFrame(
        [
            {"member_code": "minister", "office_name": "Minister for Testing", "office_start": "2026-01-01", "office_end": ""},
        ]
    )


def test_oral_question_section_counts_questions_and_speeches_once():
    sections = pd.DataFrame([{"debate_section_id": "sec-a", "show_as": "Urban Development"}])
    result = build_oral_question_sections(
        questions=_questions(), speeches=_speeches(), debate_sections=sections, member_offices=_offices(),
        source_batch_id="batch-1", contract_version=1,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["debate_section_id"] == "sec-a"
    assert row["section_heading"] == "Urban Development"
    assert row["oral_question_count"] == 2
    assert bool(row["grouped_exchange"]) is True
    assert row["related_speech_count"] == 5
    assert row["related_speaker_count"] == 4
    assert row["asking_member_count"] == 2
    assert row["participating_submitting_member_count"] == 1
    assert row["participating_submitter_share"] == 0.5
    assert row["ordinary_non_submitter_td_count"] == 1
    assert row["related_speech_word_count"] == 64
    assert row["ministerial_intervention_count"] == 1
    assert row["ministerial_word_count"] == 30
    assert row["ministerial_word_share"] == 30 / 64
    assert row["chair_intervention_count"] == 1
    assert row["chair_word_count"] == 2
    assert row["ordinary_member_intervention_count"] == 2
    assert row["ordinary_member_word_count"] == 30
    assert row["collective_or_unidentified_intervention_count"] == 1
    assert row["collective_or_unidentified_word_count"] == 2
    assert json.loads(row["question_ids_json"]) == ["q1", "q2"]


def test_written_question_does_not_create_oral_question_section():
    result = build_oral_question_sections(
        questions=_questions(), speeches=_speeches(), debate_sections=pd.DataFrame(columns=["debate_section_id", "show_as"]),
        member_offices=_offices(), source_batch_id="batch-1", contract_version=1,
    )
    assert set(result["debate_section_id"]) == {"sec-a"}
    assert "sec-written" not in set(result["debate_section_id"])


def test_exchange_participants_use_exchange_participant_role_grain():
    speeches = _speeches().copy()
    speeches = pd.concat(
        [
            speeches,
            pd.DataFrame([
                {"speech_id": "s8", "debate_date": "2026-07-01", "debate_section_id": "sec-a", "speaker_member_code": "m1", "speaker_name": "Acting Chairman (Deputy One)", "word_count": 3}
            ]),
        ],
        ignore_index=True,
    )
    result = build_oral_question_exchange_participants(
        questions=_questions(), speeches=speeches, member_offices=_offices(), source_batch_id="batch-1", contract_version=1,
    )
    assert not result.duplicated(["debate_section_id", "participant_key", "participant_role"]).any()
    m1 = result[(result["debate_section_id"] == "sec-a") & (result["member_code"] == "m1")]
    assert set(m1["participant_role"]) == {"ordinary_member", "chair"}
    assert bool(m1.loc[m1["participant_role"] == "ordinary_member", "is_recorded_submitter"].iloc[0]) is True
    collective = result[result["participant_role"] == "collective_or_unidentified"].iloc[0]
    assert collective["member_code"] == ""
    assert collective["intervention_count"] == 1


def test_exchange_participant_totals_reconcile_to_oral_section_speeches():
    result = build_oral_question_exchange_participants(
        questions=_questions(), speeches=_speeches(), member_offices=_offices(), source_batch_id="batch-1", contract_version=1,
    )
    assert int(result["intervention_count"].sum()) == 5
    assert int(result["word_count"].sum()) == 64


def test_speech_context_marks_only_speeches_in_oral_question_sections():
    result = build_speech_question_context(
        speeches=_speeches(), questions=_questions(), source_batch_id="batch-1", contract_version=1,
    ).set_index("speech_id")
    assert len(result) == 7
    for speech_id in ["s1", "s2", "s3", "s4", "s5"]:
        assert bool(result.loc[speech_id, "is_oral_question_related"]) is True
        assert result.loc[speech_id, "speech_context"] == "oral_question_exchange"
        assert result.loc[speech_id, "oral_question_count_in_section"] == 2
        assert json.loads(result.loc[speech_id, "related_question_ids_json"]) == ["q1", "q2"]
    for speech_id in ["s6", "s7"]:
        assert bool(result.loc[speech_id, "is_oral_question_related"]) is False
        assert result.loc[speech_id, "speech_context"] == "other"
        assert result.loc[speech_id, "oral_question_count_in_section"] == 0
        assert json.loads(result.loc[speech_id, "related_question_ids_json"]) == []


def test_speech_context_has_one_row_per_speech():
    result = build_speech_question_context(
        speeches=_speeches(), questions=_questions(), source_batch_id="batch-1", contract_version=1,
    )
    assert result["speech_id"].is_unique
    assert set(result["speech_id"]) == set(_speeches()["speech_id"])
