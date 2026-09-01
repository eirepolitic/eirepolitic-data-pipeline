from __future__ import annotations

import pandas as pd


def canonical_speeches(speeches: pd.DataFrame) -> pd.DataFrame:
    """Adapt `silver_speeches` to the semantic columns used by metric calculators.

    The canonical source table names the identified speaker `speaker_member_code`.
    The metric layer uses the generic event/entity name `member_code` so the same
    temporal helpers can be reused by speeches, questions, votes, and later facts.
    """
    required = {"speech_id", "debate_date", "speaker_member_code"}
    missing = sorted(required - set(speeches.columns))
    if missing:
        raise ValueError(f"silver_speeches missing required columns: {missing}")

    result = speeches.copy()
    if "member_code" in result.columns:
        existing = result["member_code"].fillna("").astype(str)
        source = result["speaker_member_code"].fillna("").astype(str)
        conflict = (existing != "") & (source != "") & (existing != source)
        if conflict.any():
            raise ValueError("speech source contains conflicting member_code and speaker_member_code values")
        result["member_code"] = result["member_code"].where(result["member_code"].notna(), result["speaker_member_code"])
    else:
        result["member_code"] = result["speaker_member_code"]
    return result
