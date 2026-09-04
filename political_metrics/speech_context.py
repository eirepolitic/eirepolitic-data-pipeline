from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


CONTEXT_VERSION = 1
ALLOWED_CONTEXTS = [
    "oral_question_exchange",
    "bill_or_legislation",
    "leaders_questions",
    "statements",
    "procedural_business",
    "motion_proceeding",
    "other",
]
PRECEDENCE = ALLOWED_CONTEXTS[:-1]

LEADERS_HEADINGS = {
    "Ceisteanna ó Cheannairí - Leaders' Questions",
    "Ceisteanna ó Cheannairí (Atógáil) - Leaders' Questions (Resumed)",
}
PROCEDURAL_HEADINGS = {
    "An tOrd Gnó - Order of Business",
    "An tOrd Gnó - Order of Business (Resumed)",
    "An tOrd Gnó (Atógáil) - Order of Business (Resumed)",
    "Ceisteanna ar Reachtaíocht a Gealladh - Questions on Promised Legislation",
    "Gnó na Dála - Business of Dáil",
}

SPEECH_CONTEXT_COLUMNS = [
    "speech_id",
    "debate_date",
    "debate_section_id",
    "speech_context",
    "evidence_method",
    "linked_entity_type",
    "linked_entity_id",
    "context_version",
    "source_batch_id",
    "calculated_at_utc",
    "contract_version",
]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in result.columns:
        result[col] = result[col].fillna("").astype(str).str.strip()
    return result


def _heading_map(debate_sections: pd.DataFrame) -> dict[str, str]:
    sections = _clean(debate_sections)
    required = {"debate_section_id", "heading", "show_as"}
    missing = sorted(required - set(sections.columns))
    if missing:
        raise ValueError(f"debate_sections missing required speech-context columns: {missing}")
    sections["section_heading"] = sections["show_as"].where(sections["show_as"].ne(""), sections["heading"])
    duplicate = sections.groupby("debate_section_id")["section_heading"].nunique()
    if int((duplicate > 1).sum()):
        raise ValueError("debate_sections contains debate_section_id values with multiple headings")
    return sections.drop_duplicates("debate_section_id", keep="last").set_index("debate_section_id")["section_heading"].to_dict()


def _statement_heading(value: str) -> bool:
    text = str(value or "")
    return (
        text.endswith(": Statements")
        or text.endswith(": Statements (Resumed)")
        or text.endswith(": Ráitis")
        or text.endswith(": Ráitis (Atógáil)")
    )


def _motion_heading(value: str) -> bool:
    text = str(value or "")
    endings = (
        ": Motion",
        ": Motion (Resumed)",
        ": Motion [Private Members]",
        ": Motion (Resumed) [Private Members]",
        ": Motions",
        ": Motions (Resumed)",
    )
    return text.endswith(endings)


def build_speech_context(
    *,
    speeches: pd.DataFrame,
    debate_sections: pd.DataFrame,
    speech_question_context: pd.DataFrame,
    bill_debate_sections: pd.DataFrame,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    required_speech = {"speech_id", "debate_date", "debate_section_id"}
    missing = sorted(required_speech - set(speeches.columns))
    if missing:
        raise ValueError(f"speeches missing required speech-context columns: {missing}")

    speech = _clean(speeches[["speech_id", "debate_date", "debate_section_id"]])
    if speech["speech_id"].duplicated().any():
        raise ValueError("speeches contains duplicate speech_id values")

    headings = _heading_map(debate_sections)
    speech["section_heading"] = speech["debate_section_id"].map(headings).fillna("")

    sqc = _clean(speech_question_context)
    required_q = {"speech_id", "speech_context"}
    missing_q = sorted(required_q - set(sqc.columns))
    if missing_q:
        raise ValueError(f"speech_question_context missing required columns: {missing_q}")
    oral_ids = set(sqc.loc[sqc["speech_context"].eq("oral_question_exchange"), "speech_id"])

    bills = _clean(bill_debate_sections)
    required_bill = {"bill_id", "debate_section_id"}
    missing_bill = sorted(required_bill - set(bills.columns))
    if missing_bill:
        raise ValueError(f"bill_debate_sections missing required columns: {missing_bill}")
    bill_counts = bills.groupby("debate_section_id")["bill_id"].nunique() if not bills.empty else pd.Series(dtype=int)
    if int((bill_counts > 1).sum()):
        raise ValueError("bill_debate_sections has more than one Bill for a certified debate section")
    bill_map = bills.drop_duplicates("debate_section_id").set_index("debate_section_id")["bill_id"].to_dict() if not bills.empty else {}

    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for row in speech.itertuples(index=False):
        context = "other"
        method = "fallback_other"
        linked_type = ""
        linked_id = ""
        heading = row.section_heading
        if row.speech_id in oral_ids:
            context = "oral_question_exchange"
            method = "certified_speech_question_context"
        elif row.debate_section_id in bill_map:
            context = "bill_or_legislation"
            method = "certified_bill_debate_section"
            linked_type = "bill"
            linked_id = bill_map[row.debate_section_id]
        elif heading in LEADERS_HEADINGS:
            context = "leaders_questions"
            method = "exact_section_heading_allowlist"
        elif _statement_heading(heading):
            context = "statements"
            method = "exact_section_heading_form"
        elif heading in PROCEDURAL_HEADINGS:
            context = "procedural_business"
            method = "exact_section_heading_allowlist"
        elif _motion_heading(heading):
            context = "motion_proceeding"
            method = "exact_section_heading_form"
        rows.append(
            {
                "speech_id": row.speech_id,
                "debate_date": row.debate_date,
                "debate_section_id": row.debate_section_id,
                "speech_context": context,
                "evidence_method": method,
                "linked_entity_type": linked_type,
                "linked_entity_id": linked_id,
                "context_version": CONTEXT_VERSION,
                "source_batch_id": source_batch_id,
                "calculated_at_utc": now,
                "contract_version": contract_version,
            }
        )
    return pd.DataFrame(rows, columns=SPEECH_CONTEXT_COLUMNS)


def audit_speech_context(
    *,
    speech_context: pd.DataFrame,
    speeches: pd.DataFrame,
    speech_question_context: pd.DataFrame,
    bill_debate_sections: pd.DataFrame,
) -> dict:
    context = _clean(speech_context)
    source = _clean(speeches)
    sqc = _clean(speech_question_context)
    bills = _clean(bill_debate_sections)

    source_ids = set(source["speech_id"])
    output_ids = set(context["speech_id"])
    duplicate_speech_ids = int(context["speech_id"].duplicated().sum())
    missing_source = len(source_ids - output_ids)
    extra_output = len(output_ids - source_ids)
    invalid_context = int((~context["speech_context"].isin(ALLOWED_CONTEXTS)).sum())

    oral_expected = set(sqc.loc[sqc["speech_context"].eq("oral_question_exchange"), "speech_id"])
    oral_actual = set(context.loc[context["speech_context"].eq("oral_question_exchange"), "speech_id"])
    oral_mismatch = len(oral_expected.symmetric_difference(oral_actual))

    bill_map = bills.drop_duplicates("debate_section_id").set_index("debate_section_id")["bill_id"].to_dict() if not bills.empty else {}
    bill_rows = context[context["speech_context"].eq("bill_or_legislation")]
    bill_link_mismatch = 0
    for row in bill_rows.itertuples(index=False):
        expected = bill_map.get(row.debate_section_id, "")
        if not expected or row.linked_entity_type != "bill" or row.linked_entity_id != expected:
            bill_link_mismatch += 1

    non_bill_with_entity = int(
        context.loc[~context["speech_context"].eq("bill_or_legislation"), "linked_entity_id"].ne("").sum()
    )

    counts = {k: int(v) for k, v in context["speech_context"].value_counts().to_dict().items()}
    checks = {
        "one_row_per_source_speech": len(context) == len(source),
        "speech_id_unique": duplicate_speech_ids == 0,
        "no_missing_source_speeches": missing_source == 0,
        "no_extra_output_speeches": extra_output == 0,
        "allowed_context_values_only": invalid_context == 0,
        "oral_question_context_exactly_agrees": oral_mismatch == 0,
        "bill_links_resolve_exactly": bill_link_mismatch == 0,
        "non_bill_rows_have_no_linked_bill": non_bill_with_entity == 0,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "row_count": int(len(context)),
        "context_counts": counts,
        "non_other_count": int((context["speech_context"] != "other").sum()),
        "other_count": int((context["speech_context"] == "other").sum()),
        "coverage_share": float((context["speech_context"] != "other").mean()) if len(context) else 0.0,
        "duplicate_speech_ids": duplicate_speech_ids,
        "missing_source_speeches": missing_source,
        "extra_output_speeches": extra_output,
        "oral_mismatch": oral_mismatch,
        "bill_link_mismatch": bill_link_mismatch,
        "non_bill_with_entity": non_bill_with_entity,
    }
