from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path}")


def main() -> int:
    replace_once(
        "extract/oireachtas/build_table.py",
        "from .client import OireachtasClient\n",
        "from .batch import current_batch_id, record_batch_table, validate_batch_id\nfrom .client import OireachtasClient\n",
    )
    replace_once(
        "extract/oireachtas/build_table.py",
        '    parser.add_argument("--publish-latest", choices=("auto", "true", "false"), default="auto", help="Control writes to processed/oireachtas_unified/latest/*. auto disables latest for mode=test and enables it otherwise.")',
        '    parser.add_argument("--publish-latest", choices=("auto", "true", "false"), default="auto", help="Write candidate production objects into an immutable batch. auto disables publishing for mode=test.")\n'
        '    parser.add_argument("--batch-id", default=os.getenv("OIREACHTAS_BATCH_ID", ""), help="Required immutable batch identifier when publishing is enabled.")',
    )
    replace_once(
        "extract/oireachtas/build_table.py",
        '    publish_latest = _set_latest_env(args)\n    schema = get_table_schema(args.table, Path(args.config))',
        '    publish_latest = _set_latest_env(args)\n'
        '    batch_id = validate_batch_id(args.batch_id) if args.batch_id else None\n'
        '    if publish_latest and not batch_id:\n'
        '        raise ValueError("--batch-id is required when candidate publishing is enabled")\n'
        '    if batch_id:\n'
        '        os.environ["OIREACHTAS_BATCH_ID"] = batch_id\n'
        '    schema = get_table_schema(args.table, Path(args.config))',
    )
    replace_once(
        "extract/oireachtas/build_table.py",
        '    result.manifest["publish_latest"] = publish_latest\n    result.manifest["latest_write_policy"] = "enabled" if publish_latest else "suppressed"\n    review_dir = write_review_bundle(table=result.table, manifest=result.manifest, schema=result.schema, dq=result.dq, sample_rows=result.rows, root=Path(args.review_root))',
        '    result.manifest["publish_latest"] = publish_latest\n'
        '    result.manifest["batch_id"] = batch_id\n'
        '    result.manifest["latest_write_policy"] = "batch_candidate" if publish_latest else "suppressed"\n'
        '    if publish_latest and batch_id:\n'
        '        candidate_keys = [\n'
        '            str(result.manifest.get("latest_csv_key") or ""),\n'
        '            str(result.manifest.get("latest_parquet_key") or ""),\n'
        '        ]\n'
        '        record_batch_table(\n'
        '            s3,\n'
        '            bucket=args.s3_bucket,\n'
        '            batch_id=batch_id,\n'
        '            table=result.table,\n'
        '            manifest=result.manifest,\n'
        '            schema=result.schema,\n'
        '            dq=result.dq,\n'
        '            candidate_keys=[key for key in candidate_keys if key],\n'
        '        )\n'
        '    review_dir = write_review_bundle(\n'
        '        table=result.table, manifest=result.manifest, schema=result.schema, dq=result.dq,\n'
        '        sample_rows=result.rows, root=Path(args.review_root), sample_limit=args.sample_rows\n'
        '    )',
    )
    replace_once(
        "extract/oireachtas/build_table.py",
        '    print(f"TABLE={result.table}\\nMODE={args.mode}\\nROWS={result.manifest.get(\'output_rows\')}\\nCOLUMNS={len(result.schema.get(\'columns\', []))}\\nPRIMARY_KEY={\',\'.join(result.schema.get(\'primary_key\', []))}\\nPRIMARY_KEY_UNIQUE={str(result.manifest.get(\'primary_key_unique\')).lower()}\\nPUBLISH_LATEST={str(publish_latest).lower()}")',
        '    print(f"TABLE={result.table}\\nMODE={args.mode}\\nROWS={result.manifest.get(\'output_rows\')}\\nCOLUMNS={len(result.schema.get(\'columns\', []))}\\nPRIMARY_KEY={\',\'.join(result.schema.get(\'primary_key\', []))}\\nPRIMARY_KEY_UNIQUE={str(result.manifest.get(\'primary_key_unique\')).lower()}\\nPUBLISH_LATEST={str(publish_latest).lower()}\\nBATCH_ID={batch_id or \"\"}")',
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
