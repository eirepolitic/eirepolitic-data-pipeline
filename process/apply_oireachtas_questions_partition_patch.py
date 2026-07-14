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
    path = "extract/oireachtas/table_questions.py"
    replace_once(
        path,
        "from .normalize import normalize_format_url, parse_iso_date, stable_hash, stable_record_hash, utc_now_iso\n",
        "from .normalize import normalize_format_url, parse_iso_date, stable_hash, stable_record_hash, utc_now_iso\nfrom .partitioned_fetch import get_date_partitioned_json_summary\n",
    )
    replace_once(
        path,
        "    summary = client.get_json_summary(endpoint, params=params)\n",
        "    summary = get_date_partitioned_json_summary(client, endpoint, params=params)\n",
    )
    replace_once(
        path,
        '        "raw_rows": len(results),\n',
        '        "raw_rows": len(results),\n        "pagination": dict(summary.pagination or {}),\n',
    )
    print("integrated adaptive date partitioning into silver_questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
