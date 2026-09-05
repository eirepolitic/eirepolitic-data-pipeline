#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
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
from lxml import etree

from extract.oireachtas.batch import batch_key_for_production_key, validate_batch_id
from political_metrics.candidate_publish import logical_metric_key, publish_dataset_to_candidate
from political_metrics.materialize import get_dataset_contract, load_materialization_contract
from political_metrics.written_question_answers import (
    ANSWER_SECTION_COLUMNS,
    QUESTION_BRIDGE_COLUMNS,
    audit_written_answer_foundations,
    build_written_answer_foundations,
)

CONTRACT_PATH = REPO_ROOT / "configs/political_metrics/written_question_answers.yml"
QUESTIONS_KEY = "processed/oireachtas_unified/latest/csv/silver_questions.csv"
SECTION_STRING_COLUMNS = [
    "debate_section_id", "answer_date", "section_heading", "answer_status", "answer_text", "respondent_ref",
    "respondent_role_ref", "observed_question_eids_json", "summary_texts_json", "source_xml_url", "source_xml_uri",
    "source_document_url", "source_document_sha256", "source_section_sha256", "source_batch_id", "calculated_at_utc",
]
SECTION_INTEGER_COLUMNS = ["observed_question_count", "embedded_table_count", "answer_version", "contract_version"]
SECTION_BOOLEAN_COLUMNS = ["grouped_answer", "referred_or_direct_reply"]
BRIDGE_STRING_COLUMNS = [
    "question_id", "debate_section_id", "question_date", "question_xml_match_status", "observed_question_eid",
    "source_xml_url", "source_batch_id", "calculated_at_utc",
]
BRIDGE_INTEGER_COLUMNS = ["bridge_version", "contract_version"]
_SECTION_XML_RE = re.compile(r"/dbsect_[^/]+\.xml$")


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if isinstance(tag, str) and "}" in tag else str(tag)


def _eid(element) -> str:
    for key, value in element.attrib.items():
        if _local(key) == "eId":
            return str(value)
    return ""


def _attr(element, name: str) -> str:
    for key, value in element.attrib.items():
        if _local(key) == name:
            return str(value)
    return ""


def _suffix(value: str) -> str:
    return str(value or "").rstrip("/").split("/")[-1]


def _daily_document_url(section_url: str) -> str:
    url = str(section_url or "")
    if not _SECTION_XML_RE.search(url):
        raise ValueError(f"unexpected Written-answer section XML URL: {url!r}")
    return _SECTION_XML_RE.sub("/main.xml", url)


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


def _fetch_documents(urls: list[str], *, workers: int, timeout_seconds: int, retries: int) -> tuple[dict[str, bytes], dict[str, str]]:
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
    *, current_groups: dict[str, dict], existing_sections: pd.DataFrame | None, existing_bridge: pd.DataFrame | None
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
        if len(urls) == 1 and section_url.get(section_id) == urls[0] and bridge_ids.get(section_id) == current["question_ids"]:
            reusable.add(section_id)
    return reusable


def _restamp(frame: pd.DataFrame, *, source_batch_id: str, contract_version: int, version_col: str) -> pd.DataFrame:
    result = frame.copy()
    result["source_batch_id"] = source_batch_id
    result["calculated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["contract_version"] = contract_version
    result[version_col] = pd.to_numeric(result[version_col], errors="raise").astype("int64")
    return result


def _normalize_sections(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=ANSWER_SECTION_COLUMNS).copy()
    for col in SECTION_STRING_COLUMNS:
        result[col] = result[col].fillna("").astype(str)
    for col in SECTION_INTEGER_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="raise").astype("int64")
    for col in SECTION_BOOLEAN_COLUMNS:
        if result[col].dtype != bool:
            result[col] = result[col].fillna(False).map(
                lambda v: v if isinstance(v, bool) else str(v).strip().lower() in {"true", "1", "yes"}
            ).astype(bool)
    return result


def _normalize_bridge(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=QUESTION_BRIDGE_COLUMNS).copy()
    for col in BRIDGE_STRING_COLUMNS:
        result[col] = result[col].fillna("").astype(str)
    for col in BRIDGE_INTEGER_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="raise").astype("int64")
    return result


def _extract_requested_sections(
    *,
    document_url: str,
    document_bytes: bytes,
    requested: dict[str, str],
) -> tuple[dict[str, bytes], dict[str, str], dict[str, str], list[str]]:
    root = etree.fromstring(document_bytes)
    by_eid: dict[str, bytes] = {}
    for element in root.iter():
        if _local(element.tag) != "debateSection":
            continue
        eid = _eid(element)
        if eid in requested and _attr(element, "name") == "writtenAnswer":
            by_eid[eid] = etree.tostring(element, encoding="utf-8")

    document_hash = hashlib.sha256(document_bytes).hexdigest()
    xml_by_url: dict[str, bytes] = {}
    document_by_url: dict[str, str] = {}
    document_hash_by_url: dict[str, str] = {}
    missing: list[str] = []
    for eid, section_url in requested.items():
        payload = by_eid.get(eid)
        if payload is None:
            missing.append(eid)
            continue
        xml_by_url[section_url] = payload
        document_by_url[section_url] = document_url
        document_hash_by_url[section_url] = document_hash
    return xml_by_url, document_by_url, document_hash_by_url, missing


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build certified Written Parliamentary Question answer foundations in one candidate batch.")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout-seconds", type=int, default=45)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--document-chunk-size", type=int, default=4)
    p.add_argument("--max-sections", type=int, default=0, help="Validation-only cap. Publishing is disabled when set.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-path", default="written_question_answers_candidate_report.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    batch_id = validate_batch_id(args.batch_id)
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8 for large daily XML documents")
    if args.document_chunk_size < 1 or args.document_chunk_size > 10:
        raise ValueError("document-chunk-size must be between 1 and 10")

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
    bad_url_sections = [section for section, item in current_groups.items() if len(item["urls"]) != 1]
    if bad_url_sections:
        raise RuntimeError(f"Written-answer sections must have exactly one source XML URL; examples={bad_url_sections[:20]}")

    existing_sections = None
    existing_bridge = None
    if not args.max_sections:
        existing_sections = _read_candidate_csv(
            s3, bucket=args.bucket, batch_id=batch_id, logical_key=logical_metric_key(section_contract, "csv"), required=False
        )
        existing_bridge = _read_candidate_csv(
            s3, bucket=args.bucket, batch_id=batch_id, logical_key=logical_metric_key(bridge_contract, "csv"), required=False
        )

    reusable = _reusable_sections(
        current_groups=current_groups, existing_sections=existing_sections, existing_bridge=existing_bridge
    )
    fetch_sections = sorted(set(current_groups) - reusable)

    sections_by_document: dict[str, dict[str, str]] = {}
    section_id_by_eid_by_document: dict[str, dict[str, str]] = {}
    for section_id in fetch_sections:
        section_url = current_groups[section_id]["urls"][0]
        document_url = _daily_document_url(section_url)
        eid = _suffix(section_id)
        sections_by_document.setdefault(document_url, {})[eid] = section_url
        section_id_by_eid_by_document.setdefault(document_url, {})[eid] = section_id

    document_urls = sorted(sections_by_document)
    new_section_frames: list[pd.DataFrame] = []
    new_bridge_frames: list[pd.DataFrame] = []
    total_fetch_failures: dict[str, str] = {}
    total_missing_sections: list[dict] = []

    for offset in range(0, len(document_urls), args.document_chunk_size):
        chunk_document_urls = document_urls[offset : offset + args.document_chunk_size]
        documents, failures = _fetch_documents(
            chunk_document_urls, workers=args.workers, timeout_seconds=args.timeout_seconds, retries=args.retries
        )
        total_fetch_failures.update(failures)
        if failures:
            raise RuntimeError(
                "Written-answer daily document fetch failed: "
                + json.dumps(dict(list(failures.items())[:20]), ensure_ascii=False)
            )

        xml_by_url: dict[str, bytes] = {}
        document_by_url: dict[str, str] = {}
        document_hash_by_url: dict[str, str] = {}
        chunk_section_ids: list[str] = []
        for document_url in chunk_document_urls:
            extracted, source_docs, doc_hashes, missing = _extract_requested_sections(
                document_url=document_url,
                document_bytes=documents[document_url],
                requested=sections_by_document[document_url],
            )
            xml_by_url.update(extracted)
            document_by_url.update(source_docs)
            document_hash_by_url.update(doc_hashes)
            chunk_section_ids.extend(section_id_by_eid_by_document[document_url].values())
            if missing:
                total_missing_sections.extend({"document_url": document_url, "section_eid": eid} for eid in missing)

        if total_missing_sections:
            raise RuntimeError(
                "Written-answer sections missing from daily document: "
                + json.dumps(total_missing_sections[:20], ensure_ascii=False)
            )

        chunk_questions = written[written["debate_section_id"].astype(str).isin(chunk_section_ids)].copy()
        sections_chunk, bridge_chunk, audit_chunk = build_written_answer_foundations(
            written_questions=chunk_questions,
            xml_by_url=xml_by_url,
            source_batch_id=batch_id,
            contract_version=contract_version,
            source_document_by_url=document_by_url,
            source_document_sha256_by_url=document_hash_by_url,
        )
        if not audit_chunk.get("ready"):
            raise RuntimeError(
                "Written-answer daily document parse gate failed: "
                + json.dumps({"document_offset": offset, "audit": audit_chunk}, ensure_ascii=False)
            )
        new_section_frames.append(sections_chunk)
        new_bridge_frames.append(bridge_chunk)
        print(json.dumps({
            "progress_documents_complete": min(offset + len(chunk_document_urls), len(document_urls)),
            "progress_documents_total": len(document_urls),
            "chunk_documents": len(chunk_document_urls),
            "chunk_sections": int(len(sections_chunk)),
            "chunk_questions": int(len(bridge_chunk)),
        }, sort_keys=True), flush=True)

    new_sections = pd.concat(new_section_frames, ignore_index=True) if new_section_frames else pd.DataFrame(columns=ANSWER_SECTION_COLUMNS)
    new_bridge = pd.concat(new_bridge_frames, ignore_index=True) if new_bridge_frames else pd.DataFrame(columns=QUESTION_BRIDGE_COLUMNS)

    if reusable:
        reused_sections = existing_sections[existing_sections["debate_section_id"].astype(str).isin(reusable)].copy()
        reused_bridge = existing_bridge[existing_bridge["debate_section_id"].astype(str).isin(reusable)].copy()
        reused_sections = _restamp(reused_sections, source_batch_id=batch_id, contract_version=contract_version, version_col="answer_version")
        reused_bridge = _restamp(reused_bridge, source_batch_id=batch_id, contract_version=contract_version, version_col="bridge_version")
        sections = pd.concat([reused_sections, new_sections], ignore_index=True)
        bridge = pd.concat([reused_bridge, new_bridge], ignore_index=True)
    else:
        sections = new_sections
        bridge = new_bridge

    sections = _normalize_sections(sections)
    bridge = _normalize_bridge(bridge)
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
            s3, bucket=args.bucket, batch_id=batch_id, frame=sections, dataset=section_contract,
            contract_version=contract_version, source_batch_id=batch_id,
        )
        published["written_question_answer_bridge"] = publish_dataset_to_candidate(
            s3, bucket=args.bucket, batch_id=batch_id, frame=bridge, dataset=bridge_contract,
            contract_version=contract_version, source_batch_id=batch_id,
        )

    report = {
        "batch_id": batch_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_written_question_count": full_written_count,
        "selected_written_question_count": int(len(written)),
        "selected_section_count": int(len(current_groups)),
        "reused_section_count": int(len(reusable)),
        "fetched_section_count": int(len(fetch_sections)),
        "unique_source_document_count": int(len(document_urls)),
        "source_document_fetch_failure_count": int(len(total_fetch_failures)),
        "missing_section_from_document_count": int(len(total_missing_sections)),
        "audit": audit,
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
