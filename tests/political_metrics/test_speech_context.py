import pandas as pd

from political_metrics.speech_context import audit_speech_context, build_speech_context


def _sections():
    return pd.DataFrame([
        {"debate_section_id":"s1","heading":"Oral","show_as":"Oral"},
        {"debate_section_id":"s2","heading":"Bill A: Second Stage","show_as":"Bill A: Second Stage"},
        {"debate_section_id":"s3","heading":"Ceisteanna ó Cheannairí - Leaders' Questions","show_as":"Ceisteanna ó Cheannairí - Leaders' Questions"},
        {"debate_section_id":"s4","heading":"Housing: Statements","show_as":"Housing: Statements"},
        {"debate_section_id":"s5","heading":"Gnó na Dála - Business of Dáil","show_as":"Gnó na Dála - Business of Dáil"},
        {"debate_section_id":"s6","heading":"Housing Supply: Motion [Private Members]","show_as":"Housing Supply: Motion [Private Members]"},
        {"debate_section_id":"s7","heading":"Other debate","show_as":"Other debate"},
    ])


def _speeches():
    return pd.DataFrame([
        {"speech_id":"p1","debate_date":"2026-01-01","debate_section_id":"s1"},
        {"speech_id":"p2","debate_date":"2026-01-01","debate_section_id":"s2"},
        {"speech_id":"p3","debate_date":"2026-01-01","debate_section_id":"s3"},
        {"speech_id":"p4","debate_date":"2026-01-01","debate_section_id":"s4"},
        {"speech_id":"p5","debate_date":"2026-01-01","debate_section_id":"s5"},
        {"speech_id":"p6","debate_date":"2026-01-01","debate_section_id":"s6"},
        {"speech_id":"p7","debate_date":"2026-01-01","debate_section_id":"s7"},
    ])


def _sqc():
    return pd.DataFrame([
        {"speech_id":"p1","speech_context":"oral_question_exchange"},
        {"speech_id":"p2","speech_context":"other"},
        {"speech_id":"p3","speech_context":"other"},
        {"speech_id":"p4","speech_context":"other"},
        {"speech_id":"p5","speech_context":"other"},
        {"speech_id":"p6","speech_context":"other"},
        {"speech_id":"p7","speech_context":"other"},
    ])


def _bills():
    return pd.DataFrame([{"bill_id":"bill-a","debate_section_id":"s2"}])


def test_build_speech_context_assigns_all_certified_contexts_and_other():
    out = build_speech_context(
        speeches=_speeches(),
        debate_sections=_sections(),
        speech_question_context=_sqc(),
        bill_debate_sections=_bills(),
        source_batch_id="batch-1",
        contract_version=1,
    )
    got = dict(zip(out.speech_id, out.speech_context))
    assert got == {
        "p1":"oral_question_exchange",
        "p2":"bill_or_legislation",
        "p3":"leaders_questions",
        "p4":"statements",
        "p5":"procedural_business",
        "p6":"motion_proceeding",
        "p7":"other",
    }
    bill = out.loc[out.speech_id.eq("p2")].iloc[0]
    assert bill.linked_entity_type == "bill"
    assert bill.linked_entity_id == "bill-a"


def test_oral_question_precedence_wins_over_heading_and_bill_context():
    sections = _sections().copy()
    sections.loc[sections.debate_section_id.eq("s1"), "show_as"] = "Housing: Statements"
    bills = pd.DataFrame([{"bill_id":"bill-x","debate_section_id":"s1"}])
    out = build_speech_context(
        speeches=_speeches().iloc[[0]],
        debate_sections=sections,
        speech_question_context=_sqc().iloc[[0]],
        bill_debate_sections=bills,
        source_batch_id="batch-1",
        contract_version=1,
    )
    assert out.iloc[0].speech_context == "oral_question_exchange"
    assert out.iloc[0].linked_entity_id == ""


def test_statement_and_motion_rules_do_not_use_broad_substrings():
    sections = pd.DataFrame([
        {"debate_section_id":"x1","heading":"Budget Statement 2026","show_as":"Budget Statement 2026"},
        {"debate_section_id":"x2","heading":"Statement of Estimates: Motion","show_as":"Statement of Estimates: Motion"},
    ])
    speeches = pd.DataFrame([
        {"speech_id":"x1p","debate_date":"2026-01-01","debate_section_id":"x1"},
        {"speech_id":"x2p","debate_date":"2026-01-01","debate_section_id":"x2"},
    ])
    sqc = pd.DataFrame([{"speech_id":"x1p","speech_context":"other"},{"speech_id":"x2p","speech_context":"other"}])
    bills = pd.DataFrame(columns=["bill_id","debate_section_id"])
    out = build_speech_context(
        speeches=speeches,
        debate_sections=sections,
        speech_question_context=sqc,
        bill_debate_sections=bills,
        source_batch_id="batch-1",
        contract_version=1,
    )
    assert dict(zip(out.speech_id,out.speech_context)) == {"x1p":"other","x2p":"motion_proceeding"}


def test_audit_requires_exact_source_coverage_and_oral_agreement():
    out = build_speech_context(
        speeches=_speeches(),
        debate_sections=_sections(),
        speech_question_context=_sqc(),
        bill_debate_sections=_bills(),
        source_batch_id="batch-1",
        contract_version=1,
    )
    audit = audit_speech_context(
        speech_context=out,
        speeches=_speeches(),
        speech_question_context=_sqc(),
        bill_debate_sections=_bills(),
    )
    assert audit["ready"] is True
    assert audit["row_count"] == 7
    assert audit["other_count"] == 1
    assert audit["oral_mismatch"] == 0
    assert audit["bill_link_mismatch"] == 0
