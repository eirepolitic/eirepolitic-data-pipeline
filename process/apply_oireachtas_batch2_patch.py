from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match in {path}, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path}")


def main() -> int:
    replace_once(
        "extract/oireachtas/build_table.py",
        '    parser.add_argument("--limit", type=int, default=25, help="API/test row limit.")',
        '    parser.add_argument("--limit", type=int, default=25, help="API page size in production; maximum output rows only in mode=test.")',
    )
    replace_once(
        "extract/oireachtas/build_table.py",
        '    common = {"client": client, "s3": s3, "bucket": args.s3_bucket, "schema": schema, "limit": args.limit, "mode": args.mode}',
        '    effective_limit = args.limit if args.mode == "test" else 2_147_483_647\n'
        '    common = {"client": client, "s3": s3, "bucket": args.s3_bucket, "schema": schema, "limit": effective_limit, "mode": args.mode}',
    )
    replace_once(
        "extract/oireachtas/build_table.py",
        '    result.manifest["publish_latest"] = publish_latest',
        '    result.manifest["requested_page_size"] = args.limit\n'
        '    result.manifest["effective_output_limit"] = args.limit if args.mode == "test" else None\n'
        '    result.manifest["publish_latest"] = publish_latest',
    )

    for path in (
        "extract/oireachtas/table_gold_current_members.py",
        "extract/oireachtas/table_gold_member_activity_yearly.py",
        "extract/oireachtas/table_gold_member_activity_monthly.py",
        "extract/oireachtas/table_gold_constituency_activity_yearly.py",
        "extract/oireachtas/table_gold_content_fact_pool.py",
        "extract/oireachtas/table_control_pipeline_runs.py",
        "extract/oireachtas/table_control_table_manifests.py",
        "extract/oireachtas/table_control_data_quality_results.py",
    ):
        text = Path(path).read_text(encoding="utf-8")
        changed = text.replace(".head(max(1, limit)).copy()", ".copy()")
        changed = changed.replace(".head(max(1, limit))", "")
        if changed != text:
            Path(path).write_text(changed, encoding="utf-8")
            print(f"removed production output cap from {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
