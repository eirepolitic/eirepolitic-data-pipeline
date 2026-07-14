from __future__ import annotations

from pathlib import Path


def main() -> int:
    path = Path("extract/oireachtas/build_table.py")
    text = path.read_text(encoding="utf-8")
    old = '''        candidate_keys = [
            str(result.manifest.get("latest_csv_key") or ""),
            str(result.manifest.get("latest_parquet_key") or ""),
        ]'''
    new = '''        candidate_keys = [
            str(result.s3_keys.get("latest_csv") or ""),
            str(result.s3_keys.get("latest_parquet") or ""),
        ]'''
    count = text.count(old)
    if count == 0 and new in text:
        print("candidate key registration already corrected")
        return 0
    if count != 1:
        raise RuntimeError(f"Expected one candidate key block, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("corrected Batch 4 candidate key registration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
