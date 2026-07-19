from __future__ import annotations

from pathlib import Path


CONFIGS = {
    "extract/oireachtas/table_member_parties.py": {
        "import_anchor": "from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json\n",
        "import_line": "from .history_dedupe import dedupe_history_rows\n",
        "old_dedupe": '    rows = _dedupe_rows(rows, primary_key="member_party_id")\n    df = pd.DataFrame(rows, columns=schema.columns)\n',
        "new_dedupe": '''    dedupe = dedupe_history_rows(
        rows,
        business_key=("member_code", "party_uri", "party_start", "party_end"),
        compared_fields=("membership_id", "party_name", "is_current"),
    )
    rows = dedupe.rows
    df = pd.DataFrame(rows, columns=schema.columns)
''',
        "dq_anchor": "    dq = _dq_results(df, schema)\n",
        "dq_replacement": '''    dq = _dq_results(df, schema)
    conflict_count = len(dedupe.conflicting_keys)
    dq["checks"].append({
        "check_name": "business_key_unique",
        "status": "pass" if conflict_count == 0 else "fail",
        "metric_value": conflict_count,
        "threshold": 0,
        "conflicting_keys": dedupe.conflicting_keys[:20],
    })
    if conflict_count:
        dq["dq_status"] = "fail"
''',
        "manifest_anchor": '        "raw_rows": len(results),\n        "output_rows": int(len(df)),\n',
        "manifest_replacement": '''        "raw_rows": len(results),
        "output_rows": int(len(df)),
        "duplicate_business_rows_removed": dedupe.duplicate_rows_removed,
        "conflicting_business_keys": dedupe.conflicting_keys,
''',
    },
    "extract/oireachtas/table_member_constituencies.py": {
        "import_anchor": "from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json\n",
        "import_line": "from .history_dedupe import dedupe_history_rows\n",
        "old_dedupe": '    rows = _dedupe_rows(rows, primary_key="member_constituency_id")\n    df = pd.DataFrame(rows, columns=schema.columns)\n',
        "new_dedupe": '''    dedupe = dedupe_history_rows(
        rows,
        business_key=("member_code", "constituency_uri", "represent_start", "represent_end"),
        compared_fields=("membership_id", "constituency_name", "is_current"),
    )
    rows = dedupe.rows
    df = pd.DataFrame(rows, columns=schema.columns)
''',
        "dq_anchor": "    dq = _dq_results(df, schema)\n",
        "dq_replacement": '''    dq = _dq_results(df, schema)
    conflict_count = len(dedupe.conflicting_keys)
    dq["checks"].append({
        "check_name": "business_key_unique",
        "status": "pass" if conflict_count == 0 else "fail",
        "metric_value": conflict_count,
        "threshold": 0,
        "conflicting_keys": dedupe.conflicting_keys[:20],
    })
    if conflict_count:
        dq["dq_status"] = "fail"
''',
        "manifest_anchor": '        "raw_rows": len(results),\n        "output_rows": int(len(df)),\n',
        "manifest_replacement": '''        "raw_rows": len(results),
        "output_rows": int(len(df)),
        "duplicate_business_rows_removed": dedupe.duplicate_rows_removed,
        "conflicting_business_keys": dedupe.conflicting_keys,
''',
    },
}


def apply() -> None:
    for file_name, cfg in CONFIGS.items():
        path = Path(file_name)
        text = path.read_text(encoding="utf-8")
        if cfg["import_line"] not in text:
            if cfg["import_anchor"] not in text:
                raise RuntimeError(f"import anchor missing in {file_name}")
            text = text.replace(cfg["import_anchor"], cfg["import_anchor"] + cfg["import_line"], 1)
        for old, new, label in (
            (cfg["old_dedupe"], cfg["new_dedupe"], "dedupe"),
            (cfg["dq_anchor"], cfg["dq_replacement"], "dq"),
            (cfg["manifest_anchor"], cfg["manifest_replacement"], "manifest"),
        ):
            if old not in text:
                raise RuntimeError(f"{label} anchor missing in {file_name}")
            text = text.replace(old, new, 1)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    apply()
