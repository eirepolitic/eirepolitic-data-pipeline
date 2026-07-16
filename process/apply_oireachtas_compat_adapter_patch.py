from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0 and new in text:
        return
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    path = "extract/oireachtas/downstream_compat.py"
    replace_once(
        path,
        'SOURCE_CURRENT_MEMBERS = "processed/oireachtas_unified/latest/csv/gold_current_members.csv"',
        'SOURCE_CURRENT_MEMBERS = "processed/oireachtas_unified/latest/csv/silver_members.csv"',
    )
    replace_once(
        path,
        '''    output["constituency"] = _col(df, "constituency_name")
    output["party"] = _col(df, "party_name")
    output["house_no"] = _col(df, "house_no")''',
        '''    output["constituency"] = _first_col(df, "constituency_name", "latest_constituency_name")
    output["party"] = _first_col(df, "party_name", "latest_party_name")
    output["house_no"] = _first_col(df, "house_no", "latest_house_no")''',
    )
    replace_once(
        path,
        '''def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), dtype="object")
''',
        '''def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), dtype="object")


def _first_col(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return _col(df, name)
    return pd.Series([""] * len(df), dtype="object")
''',
    )
    print("patched compatibility roster source and fallbacks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
