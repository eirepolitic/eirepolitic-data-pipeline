from __future__ import annotations

from pathlib import Path


def main() -> int:
    path = Path("extract/oireachtas/compat_comparison.py")
    text = path.read_text(encoding="utf-8")
    old = '''    print(json.dumps({"table": TABLE_NAME, "rows": len(result.rows), "dq_status": result.dq.get("dq_status"), "run_id": result.manifest.get("run_id")}, indent=2, sort_keys=True))'''
    new = '''    print(json.dumps({
        "table": TABLE_NAME,
        "rows": result.rows,
        "dq": result.dq,
        "dq_status": result.dq.get("dq_status"),
        "run_id": result.manifest.get("run_id"),
    }, indent=2, sort_keys=True))'''
    count = text.count(old)
    if count == 0 and new in text:
        return 0
    if count != 1:
        raise RuntimeError(f"Expected one comparison output block, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
