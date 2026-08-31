from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from extract.oireachtas.batch import current_batch_id
from extract.oireachtas.io_s3 import candidate_publishing_enabled, make_s3_client, put_dataframe_csv
from process.oireachtas_speech_issue_classifier import (
    DEFAULT_MODEL,
    ENRICHMENT_CSV_KEY,
    LEGACY_CLASSIFIED_KEY,
    SILVER_SPEECHES_KEY,
    prepare_classification_plan,
    read_s3_csv,
    validate_enrichment,
    write_outputs,
)
from process.oireachtas_speech_issue_openai import classify_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable full unified speech issue classification backfill")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ca-central-1"))
    parser.add_argument("--report-path", default="speech_issue_backfill_report.json")
    return parser.parse_args()


def write_checkpoint(s3: Any, *, bucket: str, output: pd.DataFrame) -> None:
    # During an active candidate batch the IO layer redirects this logical key
    # into the candidate. It is intentionally not registered in the manifest
    # until the dataset is complete and passes final DQ.
    put_dataframe_csv(s3, bucket=bucket, key=ENRICHMENT_CSV_KEY, df=output)


def main() -> int:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    if not current_batch_id() or not candidate_publishing_enabled():
        raise RuntimeError("Backfill requires an active candidate batch with OIREACHTAS_PUBLISH_LATEST=true")

    s3 = make_s3_client(region_name=args.region)
    silver = read_s3_csv(s3, bucket=args.bucket, key=SILVER_SPEECHES_KEY)
    existing = read_s3_csv(s3, bucket=args.bucket, key=ENRICHMENT_CSV_KEY, optional=True)
    legacy = read_s3_csv(s3, bucket=args.bucket, key=LEGACY_CLASSIFIED_KEY, optional=True)
    plan = prepare_classification_plan(silver, existing=existing, legacy=legacy)
    output = plan.rows.copy()
    pending_indices = output.index[output["classification_status"] == "pending"].tolist()

    started = time.perf_counter()
    completed_since_checkpoint = 0
    model_succeeded = 0
    model_failed = 0
    failure_rows: list[dict[str, Any]] = []
    attempt_counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_idx = {
            executor.submit(
                classify_row,
                {
                    "speech_id": output.at[idx, "speech_id"],
                    "speech_text": output.at[idx, "speech_text"],
                },
                model=args.model,
            ): idx
            for idx in pending_indices
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            attempts = str(int(result.get("attempts", 0) or 0))
            attempt_counts[attempts] = attempt_counts.get(attempts, 0) + 1

            if result.get("status") == "success":
                label = str(result["issue_label"])
                output.at[idx, "issue_label"] = label
                output.at[idx, "issue_label_source"] = "openai_model"
                output.at[idx, "model_name"] = args.model
                output.at[idx, "classification_status"] = "none" if label == "NONE" else "classified"
                output.at[idx, "classified_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
                model_succeeded += 1
            else:
                output.at[idx, "classification_status"] = "failed"
                output.at[idx, "issue_label_source"] = f"classification_error:{result.get('error', 'unknown')[:120]}"
                failure_rows.append(result)
                model_failed += 1

            completed_since_checkpoint += 1
            if completed_since_checkpoint >= args.checkpoint_every:
                write_checkpoint(s3, bucket=args.bucket, output=output)
                completed_since_checkpoint = 0
                print(
                    json.dumps(
                        {
                            "checkpoint": True,
                            "model_succeeded": model_succeeded,
                            "model_failed": model_failed,
                            "remaining_pending": int((output["classification_status"] == "pending").sum()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    write_checkpoint(s3, bucket=args.bucket, output=output)
    dq = validate_enrichment(silver, output)
    elapsed = time.perf_counter() - started
    stats = {
        **plan.stats,
        "checkpoint_resume_existing_rows": int(plan.stats.get("reused_existing", 0)),
        "model_attempted_this_run": len(pending_indices),
        "model_succeeded_this_run": model_succeeded,
        "model_failed_this_run": model_failed,
        "remaining_pending": int((output["classification_status"] == "pending").sum()),
        "attempt_counts": attempt_counts,
        "elapsed_seconds": round(elapsed, 3),
    }
    report: dict[str, Any] = {
        "batch_id": current_batch_id(),
        "model": args.model,
        "concurrency": args.concurrency,
        "checkpoint_every": args.checkpoint_every,
        "stats": stats,
        "dq": dq,
        "failure_rows": failure_rows,
        "final_manifest_written": False,
        "production_promoted": False,
    }

    if dq["pending_rows"] == 0 and dq["failed_rows"] == 0 and dq["dq_status"] == "pass":
        manifest = write_outputs(
            s3,
            bucket=args.bucket,
            silver=silver,
            enrichment=output,
            stats=stats,
            model=args.model,
        )
        report["manifest"] = manifest
        report["final_manifest_written"] = True

    Path(args.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["final_manifest_written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
