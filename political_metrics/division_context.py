from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


CONTEXT_VERSION = 1
ALLOWED_CONTEXTS = [
    "bill_or_legislation",
    "motion_proceeding",
    "procedural_business",
    "other",
]

DIVISION_CONTEXT_COLUMNS = [
    "division_id",
    "division_date",
    "debate_section_id",
    "division_context",
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


def _section_context_map(speech_context: pd.DataFrame) -> dict[str, str]:
    frame = _clean(speech_context)
    required = {"debate_section_id", "speech_context"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"speech_context missing required columns: {missing}")

    mapping: dict[str, str] = {}
    for section_id, group in frame.groupby("debate_section_id", sort=False):
        non_other = sorted(set(group.loc[group["speech_context"].ne("other"), "speech_context"]))
        if len(non_other) == 1:
            mapping[section_id] = non_other[0]
        elif len(non_other) == 0:
            mapping[section_id] = "other"
        else:
            raise ValueError(f"speech_context has multiple non-other contexts for section {section_id}: {non_other}")
    return mapping


def build_division_context(
    *,
    divisions: pd.DataFrame,
    speech_context: pd.DataFrame,
    bill_debate_sections: pd.DataFrame,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    required_div = {"division_id", "division_date", "debate_section_id"}
    missing_div = sorted(required_div - set(divisions.columns))
    if missing_div:
        raise ValueError(f"divisions missing required division-context columns: {missing_div}")

    division = _clean(divisions[["division_id", "division_date", "debate_section_id"]])
    if division["division_id"].duplicated().any():
        raise ValueError("divisions contains duplicate division_id values")

    section_context = _section_context_map(speech_context)

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
    for row in division.itertuples(index=False):
        context = "other"
        method = "fallback_other"
        linked_type = ""
        linked_id = ""

        if row.debate_section_id in bill_map:
            context = "bill_or_legislation"
            method = "certified_bill_debate_section"
            linked_type = "bill"
            linked_id = bill_map[row.debate_section_id]
        else:
            section_value = section_context.get(row.debate_section_id, "other")
            if section_value == "motion_proceeding":
                context = "motion_proceeding"
                method = "certified_speech_context_section_projection"
            elif section_value == "procedural_business":
                context = "procedural_business"
                method = "certified_speech_context_section_projection"

        rows.append(
            {
                "division_id": row.division_id,
                "division_date": row.division_date,
                "debate_section_id": row.debate_section_id,
                "division_context": context,
                "evidence_method": method,
                "linked_entity_type": linked_type,
                "linked_entity_id": linked_id,
                "context_version": CONTEXT_VERSION,
                "source_batch_id": source_batch_id,
                "calculated_at_utc": now,
                "contract_version": contract_version,
            }
        )

    return pd.DataFrame(rows, columns=DIVISION_CONTEXT_COLUMNS)


def audit_division_context(
    *,
    division_context: pd.DataFrame,
    divisions: pd.DataFrame,
    member_votes: pd.DataFrame,
    bill_debate_sections: pd.DataFrame,
) -> dict:
    context = _clean(division_context)
    source = _clean(divisions)
    votes = _clean(member_votes)
    bills = _clean(bill_debate_sections)

    source_ids = set(source["division_id"])
    output_ids = set(context["division_id"])
    duplicate_ids = int(context["division_id"].duplicated().sum())
    missing_source = len(source_ids - output_ids)
    extra_output = len(output_ids - source_ids)
    invalid_context = int((~context["division_context"].isin(ALLOWED_CONTEXTS)).sum())

    bill_map = bills.drop_duplicates("debate_section_id").set_index("debate_section_id")["bill_id"].to_dict() if not bills.empty else {}
    bill_rows = context[context["division_context"].eq("bill_or_legislation")]
    bill_link_mismatch = 0
    for row in bill_rows.itertuples(index=False):
        expected = bill_map.get(row.debate_section_id, "")
        if not expected or row.linked_entity_type != "bill" or row.linked_entity_id != expected:
            bill_link_mismatch += 1

    non_bill_with_entity = int(
        context.loc[~context["division_context"].eq("bill_or_legislation"), "linked_entity_id"].ne("").sum()
    )

    if context.empty:
        joined_votes = votes.iloc[0:0].copy()
    else:
        joined_votes = votes.merge(
            context[["division_id", "division_context"]],
            on="division_id",
            how="left",
            validate="many_to_one",
        )
    vote_join_no_multiplication = len(joined_votes) == len(votes)
    vote_join_no_missing_context = int(joined_votes["division_context"].fillna("").eq("").sum()) == 0 if len(joined_votes) else True

    counts = {k: int(v) for k, v in context["division_context"].value_counts().to_dict().items()}
    checks = {
        "one_row_per_source_division": len(context) == len(source),
        "division_id_unique": duplicate_ids == 0,
        "no_missing_source_divisions": missing_source == 0,
        "no_extra_output_divisions": extra_output == 0,
        "allowed_context_values_only": invalid_context == 0,
        "bill_links_resolve_exactly": bill_link_mismatch == 0,
        "non_bill_rows_have_no_linked_bill": non_bill_with_entity == 0,
        "member_vote_join_no_multiplication": vote_join_no_multiplication,
        "member_vote_join_has_context": vote_join_no_missing_context,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "row_count": int(len(context)),
        "context_counts": counts,
        "duplicate_division_ids": duplicate_ids,
        "missing_source_divisions": missing_source,
        "extra_output_divisions": extra_output,
        "bill_link_mismatch": bill_link_mismatch,
        "non_bill_with_entity": non_bill_with_entity,
        "member_vote_rows": int(len(votes)),
        "member_vote_rows_after_join": int(len(joined_votes)),
    }
