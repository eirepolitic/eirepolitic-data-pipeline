import hashlib

import pandas as pd

from political_metrics.written_question_answers import (
    audit_written_answer_foundations,
    build_written_answer_foundations,
    parse_written_answer_xml,
)


def _xml(*, section="dbsect_10", question_eids=("pq_1",), speech=True, summary="", reply_not_received=False, table=False):
    questions = "".join(
        f'<question eId="{qid}" by="#asker" to="#minister"><p>Question {qid}</p></question>'
        for qid in question_eids
    )
    speech_xml = '<speech eId="spk_1" by="#minister-person" as="#minister-role"><p>This is the official answer.</p></speech>' if speech else ""
    summary_xml = f"<summary>{summary}</summary>" if summary else ""
    missing_xml = "<p>Reply not received from Department.</p>" if reply_not_received else ""
    table_xml = "<table><tr><td>1</td></tr></table>" if table else ""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
      <debate>
        <debateBody>
          <debateSection eId="{section}" name="writtenAnswer">
            <heading>Sample heading</heading>
            {summary_xml}
            {questions}
            {speech_xml}
            {missing_xml}
            {table_xml}
          </debateSection>
        </debateBody>
      </debate>
    </akomaNtoso>'''.encode()


def _questions(rows):
    return pd.DataFrame(rows)


def test_parse_single_written_answer():
    parsed = parse_written_answer_xml(_xml())
    assert parsed.debate_section_eid == "dbsect_10"
    assert parsed.answer_status == "ministerial_reply_present"
    assert parsed.answer_text == "This is the official answer."
    assert parsed.respondent_ref == "#minister-person"
    assert parsed.respondent_role_ref == "#minister-role"
    assert parsed.observed_question_eids == ("pq_1",)
    assert parsed.grouped_answer is False


def test_parse_grouped_answer_and_table():
    parsed = parse_written_answer_xml(
        _xml(question_eids=("pq_1", "pq_2"), summary="Question No. 2 answered with Question No. 1.", table=True)
    )
    assert parsed.grouped_answer is True
    assert parsed.observed_question_eids == ("pq_1", "pq_2")
    assert parsed.embedded_table_count == 1


def test_parse_reply_not_received_as_source_status():
    parsed = parse_written_answer_xml(_xml(speech=False, reply_not_received=True))
    assert parsed.answer_status == "reply_not_received"
    assert parsed.answer_text == ""


def test_build_section_and_bridge_preserves_question_verification_status():
    url = "https://data.oireachtas.ie/example/dbsect_10.xml"
    daily_url = "https://data.oireachtas.ie/example/main.xml"
    xml = _xml(question_eids=("pq_1",))
    questions = _questions([
        {
            "question_id": "https://data.oireachtas.ie/ie/oireachtas/question/2026-01-01/pq_1",
            "question_date": "2026-01-01",
            "question_type": "Written",
            "debate_section_id": "https://data.oireachtas.ie/ie/oireachtas/debateSection/2026-01-01/dbsect_10",
            "source_xml_url": url,
            "source_xml_uri": "https://data.oireachtas.ie/example/dbsect_10.xml",
        },
        {
            "question_id": "https://data.oireachtas.ie/ie/oireachtas/question/2026-01-01/pq_999",
            "question_date": "2026-01-01",
            "question_type": "Written",
            "debate_section_id": "https://data.oireachtas.ie/ie/oireachtas/debateSection/2026-01-01/dbsect_10",
            "source_xml_url": url,
            "source_xml_uri": "https://data.oireachtas.ie/example/dbsect_10.xml",
        },
    ])
    sections, bridge, audit = build_written_answer_foundations(
        written_questions=questions,
        xml_by_url={url: xml},
        source_batch_id="batch-1",
        contract_version=1,
        source_document_by_url={url: daily_url},
        source_document_sha256_by_url={url: "doc-hash"},
    )
    assert len(sections) == 1
    assert len(bridge) == 2
    assert sections.iloc[0].source_section_sha256 == hashlib.sha256(xml).hexdigest()
    assert sections.iloc[0].source_document_url == daily_url
    assert sections.iloc[0].source_document_sha256 == "doc-hash"
    status = dict(zip(bridge.question_id, bridge.question_xml_match_status))
    assert status[questions.iloc[0].question_id] == "question_id_matched_in_xml"
    assert status[questions.iloc[1].question_id] == "section_matched_question_id_unmatched"
    assert audit["ready"] is True
    assert audit["question_id_unmatched_bridge_rows"] == 1


def test_section_eid_mismatch_fails_certification():
    url = "https://data.oireachtas.ie/example/dbsect_10.xml"
    questions = _questions([
        {
            "question_id": "https://data.oireachtas.ie/ie/oireachtas/question/2026-01-01/pq_1",
            "question_date": "2026-01-01",
            "question_type": "Written",
            "debate_section_id": "https://data.oireachtas.ie/ie/oireachtas/debateSection/2026-01-01/dbsect_11",
            "source_xml_url": url,
            "source_xml_uri": url,
        }
    ])
    sections, bridge, audit = build_written_answer_foundations(
        written_questions=questions,
        xml_by_url={url: _xml(section="dbsect_10")},
        source_batch_id="batch-1",
        contract_version=1,
    )
    assert sections.empty
    assert bridge.empty
    assert audit["ready"] is False
    assert audit["parse_failure_count"] == 1


def test_duplicate_bridge_is_rejected():
    questions = _questions([
        {"question_id": "q1", "question_type": "Written"},
        {"question_id": "q2", "question_type": "Written"},
    ])
    sections = pd.DataFrame([
        {"debate_section_id": "s1", "answer_status": "ministerial_reply_present", "grouped_answer": False, "referred_or_direct_reply": False, "embedded_table_count": 0}
    ])
    bridge = pd.DataFrame([
        {"question_id": "q1", "debate_section_id": "s1", "question_xml_match_status": "question_id_matched_in_xml"},
        {"question_id": "q1", "debate_section_id": "s1", "question_xml_match_status": "question_id_matched_in_xml"},
    ])
    audit = audit_written_answer_foundations(
        written_questions=questions,
        answer_sections=sections,
        question_bridge=bridge,
    )
    assert audit["ready"] is False
    assert audit["bridge_duplicate_rows"] == 1
