#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3
import pandas as pd
import requests

from extract.oireachtas.batch import batch_key_for_production_key, validate_batch_id
from political_metrics.candidate_publish import publish_dataset_to_candidate
from political_metrics.materialize import get_dataset_contract, load_materialization_contract
from political_metrics.written_question_answers import (
    ANSWER_SECTION_COLUMNS,
    QUESTION_BRIDGE_COLUMNS,
    audit_written_answer_foundations,
    build_written_answer_foundations,
)

CONTRACT_PATH = REPO_ROOT / "configs/political_metrics/written_question_answers.yml"
QUESTIONS_KEY = "processed/oireachtas_unified/latest/csv/silver_questions.csv"


def _logical_csv_key(dataset: dict) -> str:
    prefix = str(dataset["output_prefix"]).rstrip("/")
    name = str(dataset["dataset_name"])
    return f"{prefix}/csv/{name}.csv"


def _read_candidate_csv(s3, *, bucket: str, batch_id: str, logical_key: str, required: bool = True) -> pd.DataFrame | None:
    key = batch_key_for_production_key(logical_key, batch_id)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception:
        if required:
            raise
        return None
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""])


def _fetch_one(url: str, *, timeout_seconds: int, retries: int) -> tuple[str, bytes | None, str]:
    headers = {"User-Agent": "EirePolitic Written PQ answer ingestion/1.0"}
    last = ""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=(5, timeout_seconds), headers=headers)
            response.raise_for_status()
            return url, response.content, ""
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))
    return url, None, last


def _fetch_xml(urls: list[str], *, workers: int, timeout_seconds: int, retries: int) -> tuple[dict[str, bytes], dict[str, str]]:
    found: dict[str, bytes] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_one, url, timeout_seconds=timeout_seconds, retries=retries) for url in urls]
        for future in as_completed(futures):
            url, content, error = future.result()
            if content is not None:
                found[url] = content
            else:
                failures[url] = error
    return found, failures


def _current_groups(questions: pd.DataFrame) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for section_id, group in questions.groupby("debate_section_id", dropna=False, sort=False):
        section_id = str(section_id or "")
        if not section_id:
            continue
        urls = sorted({str(v) for v in group["source_xml_url"].dropna().tolist() if str(v)})
        groups[section_id] = {
            "urls": urls,
            "question_ids": sorted(str(v) for v in group["question_id"].dropna().tolist() if str(v)),
        }
    return groups


def _reusable_sections(
    *,
    current_groups: dict[str, dict],
    existing_sections: pd.DataFrame | None,
    existing_bridge: pd.DataFrame | None,
) -> set[str]:
    if existing_sections is None or existing_bridge is None or existing_sections.empty or existing_bridge.empty:
        return set()
    section_url = {
        str(row.debate_section_id): str(row.source_xml_url or "")
        for row in existing_sections[["debate_section_id", "source_xml_url"]].itertuples(index=False)
    }
    bridge_ids = {
        str(section): sorted(str(v) for v in group["question_id"].dropna().tolist() if str(v))
        for section, group in existing_bridge.groupby("debate_section_id", dropna=False)
    }
    reusable: set[str] = set()
    for section_id, current in current_groups.items():
        urls = current["urls"]
        if len(urls) != 1:
            continue
        if section_url.get(section_id) != urls[0]:
            continue
        if bridge_ids.get(section_id) != current["question_ids"]:
            continue
        reusable.add(section_id)
    return reusable


def _restamp(frame: pd.DataFrame, *, source_batch_id: str, contract_version: int, version_col: str) -> pd.DataFrame:
    result = frame.copy()
    result["source_batch_id"] = source_batch_id
    result["calculated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["contract_version"] = contract_version
    result[version_col] = pd.to_numeric(result[version_col], errors="raise").astype("int64")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build certified Written Parliamentary Question answer foundations in one candidate batch.")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout-seconds", type=int, default=20)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-sections", type=int, default=0, help="Validation-only cap. Publishing is disabled when set.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-path", default="written_question_answers_candidate_report.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    batch_id = validate_batch_id(args.batch_id)
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be between 1 and 16")
    s3 = boto3.client("s3", region_name=args.region)
    contract = load_materialization_contract(CONTRACT_PATH)
    contract_version = int(contract["contract_version"])
    section_contract = get_dataset_contract(contract, "written_question_answer_sections")
    bridge_contract = get_dataset_contract(contract, "written_question_answer_bridge")

    questions = _read_candidate_csv(s3, bucket=args.bucket, batch_id=batch_id, logical_key=QUESTIONS_KEY, required=True)
    written = questions[questions["question_type"].fillna("").astype(str).str.strip().str.lower().eq("written")].copy()
    written = written.sort_values(["question_date", "question_id"]).reset_index(drop=True)
    full_written_count = int(len(written))
    if args.max_sections:
        section_ids = written["debate_section_id"].dropna().astype(str).drop_duplicates().head(args.max_sections).tolist()
        written = written[written["debate_section_id"].astype(str).isin(section_ids)].copy()

    current_groups = _current_groups(written)
    if any(len(item["urls"]) != 1 for item in current_groups.values()):
        bad = [section for section, item in current_groups.items() if len(item["urls"]) != 1][:20]
        raise RuntimeError(f"Written-answer sections must have exactly one source XML URL; examples={bad}")

    existing_sections = None
    existing_bridge = None
    if not args.max_sections:
        existing_sections = _read_candidate_csv(
            s3, bucket=args.bucket, batch_id=batch_id, logical_key=_logical_csv_key(section_contract), required=False
        )
        existing_bridge = _read_candidate_csv(
            s3, bucket=args.bucket, batch_id=batch_id, logical_key=_logical_csv_key(bridge_contract), required=False
        )

    reusable = _reusable_sections(
        current_groups=current_groups,
        existing_sections=existing_sections,
        existing_bridge=existing_bridge,
    )
    fetch_sections = sorted(set(current_groups) - reusable)
    fetch_questions = written[written["debate_section_id"].astype(str).isin(fetch_sections)].copy()
    urls = sorted({current_groups[section]["urls"][0] for section in fetch_sections})
    xml_by_url, fetch_failures = _fetch_xml(
        urls, workers=args.workers, timeout_seconds=args.timeout_seconds, retries=args.retries
    )

    new_sections, new_bridge, new_audit = build_written_answer_foundations(
        written_questions=fetch_questions,
        xml_by_url=xml_by_url,
        source_batch_id=batch_id,
        contract_version=contract_version,
    )
    if fetch_failures or not new_audit.get("ready"):
        raise RuntimeError(
            "Written-answer fetch/parse gate failed: "
            + json.dumps({"fetch_failures": dict(list(fetch_failures.items())[:20]), "audit": new_audit}, ensure_ascii=False)
        )

    if reusable:
        reused_sections = existing_sections[existing_sections["debate_section_id"].astype(str).isin(reusable)].copy()
        reused_bridge = existing_bridge[existing_bridge["debate_section_id"].astype(str).isin(reusable)].copy()
        reused_sections = _restamp(
            reused_sections, source_batch_id=batch_id, contract_version=contract_version, version_col="answer_version"
        )
        reused_bridge = _restamp(
            reused_bridge, source_batch_id=batch_id, contract_version=contract_version, version_col="bridge_version"
        )
        sections = pd.concat([reused_sections, new_sections], ignore_index=True)
        bridge = pd.concat([reused_bridge, new_bridge], ignore_index=True)
    else:
        sections = new_sections
        bridge = new_bridge

    sections = sections.reindex(columns=ANSWER_SECTION_COLUMNS)
    bridge = bridge.reindex(columns=QUESTION_BRIDGE_COLUMNS)
    audit = audit_written_answer_foundations(
        written_questions=written,
        answer_sections=sections,
        question_bridge=bridge,
        parse_failures=[],
    )
    if not audit.get("ready"):
        raise RuntimeError(f"Written-answer final reconciliation gate failed: {audit}")

    publish_allowed = not args.dry_run and args.max_sections == 0
    published = {}
    if publish_allowed:
        published["written_question_answer_sections"] = publish_dataset_to_candidate(
            s3,
            bucket=args.bucket,
            batch_id=batch_id,
            frame=sections,
            dataset=section_contract,
            contract_version=contract_version,
            source_batch_id=batch_id,
        )
        published["written_question_answer_bridge"] = publish_dataset_to_candidate(
            s3,
            bucket=args.bucket,
            batch_id=batch_id,
            frame=bridge,
            dataset=bridge_contract,
            contract_version=contract_version,
            source_batch_id=batch_id,
        )

    report = {
        "batch_id": batch_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_written_question_count": full_written_count,
        "selected_written_question_count": int(len(written)),
        "selected_section_count": int(len(current_groups)),
        "reused_section_count": int(len(reusable)),
        "fetched_section_count": int(len(fetch_sections)),
        "unique_xml_fetch_count": int(len(urls)),
        "fetch_failure_count": int(len(fetch_failures)),
        "audit": audit,
        "new_section_audit": new_audit,
        "publish_allowed": publish_allowed,
        "production_pointer_changed": False,
        "published": {
            name: {"entry_name": result["entry_name"], "row_count": result["row_count"], "objects": result["objects"]}
            for name, result in published.items()
        },
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
