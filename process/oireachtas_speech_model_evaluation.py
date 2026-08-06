from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from process.oireachtas_speech_issue_classifier import (
    ISSUE_CATEGORIES,
    ISSUE_CATEGORY_SET,
    PROMPT_VERSION,
    TAXONOMY_VERSION,
    custom_id_for_speech,
    parse_batch_output_jsonl,
    structured_response_body,
    validate_label,
)

FIXTURE_REQUIRED_COLUMNS = [
    "sample_id",
    "speech_text",
    "expected_issue_label",
    "review_status",
    "reviewer",
    "reviewed_at_utc",
    "review_notes",
]
RESULT_REQUIRED_COLUMNS = [
    "sample_id",
    "model_name",
    "predicted_issue_label",
    "classification_status",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "batch_id",
    "batch_status",
    "error",
]
REVIEW_STATUSES = {"pending", "approved", "rejected"}
SUCCESS_BATCH_STATUSES = {"completed", "success"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture(path: str | Path, *, require_reviewed: bool = False) -> pd.DataFrame:
    resolved = Path(path)
    frame = pd.read_csv(resolved, dtype=str, keep_default_na=False)
    missing = sorted(set(FIXTURE_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation fixture is missing columns: {missing}")
    frame = frame[FIXTURE_REQUIRED_COLUMNS].copy()
    for column in FIXTURE_REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    if frame.empty:
        raise ValueError("Evaluation fixture is empty")
    if frame["sample_id"].eq("").any() or frame["sample_id"].duplicated().any():
        raise ValueError("Evaluation fixture requires unique, non-blank sample_id values")
    if frame["speech_text"].eq("").any():
        raise ValueError("Evaluation fixture contains blank speech_text values")
    frame["expected_issue_label"] = frame["expected_issue_label"].map(validate_label)
    invalid_review = sorted(set(frame["review_status"]) - REVIEW_STATUSES)
    if invalid_review:
        raise ValueError(f"Invalid review_status values: {invalid_review}")
    approved = frame["review_status"].eq("approved")
    approved_missing_reviewer = approved & frame["reviewer"].eq("")
    approved_missing_date = approved & frame["reviewed_at_utc"].eq("")
    if approved_missing_reviewer.any() or approved_missing_date.any():
        raise ValueError("Approved fixture rows require reviewer and reviewed_at_utc")
    if require_reviewed and not approved.all():
        counts = frame["review_status"].value_counts().to_dict()
        raise ValueError(
            "Paid evaluation requires every fixture row to be approved; "
            f"review_status_counts={counts}"
        )
    return frame


def fixture_report(path: str | Path, *, require_reviewed: bool = False) -> dict[str, Any]:
    resolved = Path(path)
    frame = load_fixture(resolved, require_reviewed=require_reviewed)
    support = {
        category: int(frame["expected_issue_label"].eq(category).sum())
        for category in ISSUE_CATEGORIES
    }
    return {
        "fixture_path": str(resolved),
        "fixture_sha256": sha256_file(resolved),
        "rows": int(len(frame)),
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "review_status_counts": frame["review_status"].value_counts().to_dict(),
        "category_support": support,
        "missing_categories": [category for category, count in support.items() if count == 0],
        "fully_reviewed": bool(frame["review_status"].eq("approved").all()),
    }


def build_evaluation_requests(
    fixture: pd.DataFrame,
    *,
    model: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    model_id = str(model or "").strip()
    if not model_id:
        raise ValueError("model is required")
    mapping_rows: list[dict[str, str]] = []
    requests: list[dict[str, Any]] = []
    for row in fixture.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        custom_id = custom_id_for_speech(f"evaluation:{sample_id}")
        mapping_rows.append(
            {
                "custom_id": custom_id,
                "sample_id": sample_id,
                "expected_issue_label": str(row["expected_issue_label"]),
            }
        )
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": structured_response_body(
                    model=model_id,
                    speech_text=str(row["speech_text"]),
                ),
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    if mapping["custom_id"].duplicated().any():
        raise ValueError("Evaluation custom_id collision")
    return mapping, requests


def prepare_evaluation_files(
    *,
    fixture_path: str | Path,
    models: Sequence[str],
    output_dir: str | Path,
    require_reviewed: bool,
) -> dict[str, Any]:
    fixture_file = Path(fixture_path)
    fixture = load_fixture(fixture_file, require_reviewed=require_reviewed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model).strip()
        if not model_id:
            continue
        safe_model = "".join(character if character.isalnum() or character in "._-" else "_" for character in model_id)
        mapping, requests = build_evaluation_requests(fixture, model=model_id)
        mapping_path = destination / f"{safe_model}.mapping.csv"
        requests_path = destination / f"{safe_model}.requests.jsonl"
        mapping.to_csv(mapping_path, index=False)
        requests_path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in requests),
            encoding="utf-8",
        )
        artifacts.append(
            {
                "model_name": model_id,
                "mapping_path": str(mapping_path),
                "mapping_sha256": sha256_file(mapping_path),
                "requests_path": str(requests_path),
                "requests_sha256": sha256_file(requests_path),
                "request_rows": len(requests),
            }
        )
    if len(artifacts) < 2:
        raise ValueError("Evaluation preparation requires at least two model IDs")
    manifest = {
        "fixture_path": str(fixture_file),
        "fixture_sha256": sha256_file(fixture_file),
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "review_required": bool(require_reviewed),
        "rows": int(len(fixture)),
        "models": artifacts,
    }
    manifest_path = destination / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}


def convert_batch_results(
    *,
    mapping_path: str | Path,
    batch_output_path: str | Path,
    model_name: str,
    batch_id: str,
    batch_status: str,
    latency_seconds: float,
    output_path: str | Path,
) -> dict[str, Any]:
    mapping = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
    required_mapping = {"custom_id", "sample_id", "expected_issue_label"}
    missing = sorted(required_mapping - set(mapping.columns))
    if missing:
        raise ValueError(f"Mapping file is missing columns: {missing}")
    parsed = parse_batch_output_jsonl(Path(batch_output_path).read_bytes())
    result_by_custom_id = {result.custom_id: result for result in parsed}
    unknown = sorted(set(result_by_custom_id) - set(mapping["custom_id"]))
    if unknown:
        raise ValueError(f"Batch output contains unknown custom IDs: {unknown[:10]}")
    rows: list[dict[str, Any]] = []
    for row in mapping.to_dict(orient="records"):
        result = result_by_custom_id.get(str(row["custom_id"]))
        rows.append(
            {
                "sample_id": row["sample_id"],
                "model_name": model_name,
                "predicted_issue_label": result.label if result else "",
                "classification_status": result.status if result else "failed",
                "input_tokens": result.input_tokens if result else "",
                "output_tokens": result.output_tokens if result else "",
                "latency_seconds": float(latency_seconds),
                "batch_id": batch_id,
                "batch_status": batch_status,
                "error": result.error if result else "Missing result from completed batch",
            }
        )
    output = pd.DataFrame(rows, columns=RESULT_REQUIRED_COLUMNS)
    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(resolved_output, index=False)
    return {
        "output_path": str(resolved_output),
        "output_sha256": sha256_file(resolved_output),
        "rows": int(len(output)),
        "model_name": model_name,
        "batch_id": batch_id,
        "batch_status": batch_status,
    }


def load_result_file(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(RESULT_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Model result file is missing columns: {missing}")
    frame = frame[RESULT_REQUIRED_COLUMNS].copy()
    for column in ("sample_id", "model_name", "predicted_issue_label", "classification_status", "batch_id", "batch_status", "error"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    if frame.empty:
        raise ValueError("Model result file is empty")
    if frame["sample_id"].eq("").any() or frame["sample_id"].duplicated().any():
        raise ValueError("Each model result file requires unique, non-blank sample_id values")
    model_names = sorted(set(frame["model_name"]))
    if len(model_names) != 1 or not model_names[0]:
        raise ValueError(f"Each result file must contain exactly one model_name: {model_names}")
    for column in ("input_tokens", "output_tokens", "latency_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _cost_projection(
    frame: pd.DataFrame,
    *,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    backfill_rows: int,
    recurring_rows: int,
) -> dict[str, Any]:
    avg_input = _safe_float(frame["input_tokens"].mean())
    avg_output = _safe_float(frame["output_tokens"].mean())
    if input_price_per_million is None or output_price_per_million is None or avg_input is None or avg_output is None:
        return {
            "pricing_supplied": False,
            "average_input_tokens": avg_input,
            "average_output_tokens": avg_output,
            "evaluation_cost": None,
            "estimated_backfill_cost": None,
            "estimated_recurring_cost": None,
        }
    per_row = avg_input * input_price_per_million / 1_000_000 + avg_output * output_price_per_million / 1_000_000
    return {
        "pricing_supplied": True,
        "input_price_per_million": input_price_per_million,
        "output_price_per_million": output_price_per_million,
        "average_input_tokens": avg_input,
        "average_output_tokens": avg_output,
        "estimated_cost_per_row": per_row,
        "evaluation_cost": per_row * len(frame),
        "estimated_backfill_cost": per_row * backfill_rows,
        "estimated_recurring_cost": per_row * recurring_rows,
        "backfill_rows": backfill_rows,
        "recurring_rows": recurring_rows,
    }


def evaluate_model_result(
    fixture: pd.DataFrame,
    result: pd.DataFrame,
    *,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
    backfill_rows: int = 0,
    recurring_rows: int = 0,
) -> dict[str, Any]:
    joined = fixture[["sample_id", "expected_issue_label"]].merge(
        result,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if joined["model_name"].isna().any():
        missing = joined.loc[joined["model_name"].isna(), "sample_id"].tolist()
        raise ValueError(f"Result file is missing fixture sample IDs: {missing[:10]}")
    extra = sorted(set(result["sample_id"]) - set(fixture["sample_id"]))
    if extra:
        raise ValueError(f"Result file contains unknown sample IDs: {extra[:10]}")

    valid_output = joined["predicted_issue_label"].isin(ISSUE_CATEGORY_SET)
    classified = joined["classification_status"].eq("classified")
    correct = valid_output & classified & joined["predicted_issue_label"].eq(joined["expected_issue_label"])
    expected_none = joined["expected_issue_label"].eq("NONE")
    predicted_none = joined["predicted_issue_label"].eq("NONE") & classified
    true_none = expected_none & predicted_none

    per_category: dict[str, Any] = {}
    for category in ISSUE_CATEGORIES:
        category_mask = joined["expected_issue_label"].eq(category)
        support = int(category_mask.sum())
        true_positive = int((category_mask & joined["predicted_issue_label"].eq(category) & classified).sum())
        predicted_count = int((joined["predicted_issue_label"].eq(category) & classified).sum())
        per_category[category] = {
            "support": support,
            "correct": true_positive,
            "recall": true_positive / support if support else None,
            "precision": true_positive / predicted_count if predicted_count else None,
        }

    batch_statuses = sorted(set(joined["batch_status"].astype(str)))
    batch_success = bool(batch_statuses) and all(status in SUCCESS_BATCH_STATUSES for status in batch_statuses)
    latency = joined["latency_seconds"].dropna()
    model_name = str(joined["model_name"].iloc[0])
    return {
        "model_name": model_name,
        "rows": int(len(joined)),
        "overall_accuracy": float(correct.mean()),
        "none_precision": int(true_none.sum()) / int(predicted_none.sum()) if predicted_none.any() else 0.0,
        "none_recall": int(true_none.sum()) / int(expected_none.sum()) if expected_none.any() else 0.0,
        "invalid_output_rate": float((~valid_output).mean()),
        "classification_failure_rate": float((~classified).mean()),
        "batch_completion_success": batch_success,
        "batch_statuses": batch_statuses,
        "input_tokens_total": int(joined["input_tokens"].fillna(0).sum()),
        "output_tokens_total": int(joined["output_tokens"].fillna(0).sum()),
        "latency_seconds": {
            "mean": _safe_float(latency.mean()),
            "p50": _safe_float(latency.quantile(0.50)) if not latency.empty else None,
            "p95": _safe_float(latency.quantile(0.95)) if not latency.empty else None,
            "max": _safe_float(latency.max()),
        },
        "cost": _cost_projection(
            joined,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            backfill_rows=backfill_rows,
            recurring_rows=recurring_rows,
        ),
        "per_category": per_category,
    }


def pairwise_agreement(result_frames: Sequence[pd.DataFrame]) -> list[dict[str, Any]]:
    agreements: list[dict[str, Any]] = []
    for left, right in combinations(result_frames, 2):
        left_model = str(left["model_name"].iloc[0])
        right_model = str(right["model_name"].iloc[0])
        joined = left[["sample_id", "predicted_issue_label"]].merge(
            right[["sample_id", "predicted_issue_label"]],
            on="sample_id",
            suffixes=("_left", "_right"),
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            raise ValueError(f"Models {left_model} and {right_model} share no sample IDs")
        agreements.append(
            {
                "left_model": left_model,
                "right_model": right_model,
                "shared_rows": int(len(joined)),
                "agreement_rate": float(
                    joined["predicted_issue_label_left"].eq(joined["predicted_issue_label_right"]).mean()
                ),
            }
        )
    return agreements


def load_pricing(path: str | Path | None) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"model_name", "input_price_per_million", "output_price_per_million"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pricing file is missing columns: {missing}")
    pricing: dict[str, tuple[float, float]] = {}
    for row in frame.to_dict(orient="records"):
        model = str(row["model_name"]).strip()
        if not model or model in pricing:
            raise ValueError("Pricing file requires unique, non-blank model_name values")
        pricing[model] = (
            float(row["input_price_per_million"]),
            float(row["output_price_per_million"]),
        )
    return pricing


def compare_results(
    *,
    fixture_path: str | Path,
    result_paths: Sequence[str | Path],
    pricing_path: str | Path | None,
    backfill_rows: int,
    recurring_rows: int,
    require_reviewed: bool,
) -> dict[str, Any]:
    fixture_file = Path(fixture_path)
    fixture = load_fixture(fixture_file, require_reviewed=require_reviewed)
    results = [load_result_file(path) for path in result_paths]
    if len(results) < 2:
        raise ValueError("Comparison requires at least two model result files")
    models = [str(frame["model_name"].iloc[0]) for frame in results]
    if len(models) != len(set(models)):
        raise ValueError(f"Duplicate model result files: {models}")
    pricing = load_pricing(pricing_path)
    evaluations: list[dict[str, Any]] = []
    for frame in results:
        model = str(frame["model_name"].iloc[0])
        model_pricing = pricing.get(model)
        evaluations.append(
            evaluate_model_result(
                fixture,
                frame,
                input_price_per_million=model_pricing[0] if model_pricing else None,
                output_price_per_million=model_pricing[1] if model_pricing else None,
                backfill_rows=backfill_rows,
                recurring_rows=recurring_rows,
            )
        )
    return {
        "fixture": fixture_report(fixture_file, require_reviewed=require_reviewed),
        "models": evaluations,
        "pairwise_agreement": pairwise_agreement(results),
        "pricing_path": str(pricing_path) if pricing_path else None,
        "backfill_rows": int(backfill_rows),
        "recurring_rows": int(recurring_rows),
        "selection_status": "requires_human_approval",
        "selected_model": None,
    }


def _model_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Oireachtas speech issue model evaluation")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-fixture")
    validate.add_argument("--fixture", required=True)
    validate.add_argument("--require-reviewed", action="store_true")

    prepare = commands.add_parser("prepare-requests")
    prepare.add_argument("--fixture", required=True)
    prepare.add_argument("--models", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--require-reviewed", action="store_true")

    convert = commands.add_parser("convert-batch-results")
    convert.add_argument("--mapping", required=True)
    convert.add_argument("--batch-output", required=True)
    convert.add_argument("--model", required=True)
    convert.add_argument("--batch-id", required=True)
    convert.add_argument("--batch-status", required=True)
    convert.add_argument("--latency-seconds", required=True, type=float)
    convert.add_argument("--output", required=True)

    compare = commands.add_parser("compare-results")
    compare.add_argument("--fixture", required=True)
    compare.add_argument("--results", action="append", required=True)
    compare.add_argument("--pricing")
    compare.add_argument("--backfill-rows", type=int, default=0)
    compare.add_argument("--recurring-rows", type=int, default=0)
    compare.add_argument("--require-reviewed", action="store_true")
    compare.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-fixture":
        report = fixture_report(args.fixture, require_reviewed=args.require_reviewed)
    elif args.command == "prepare-requests":
        report = prepare_evaluation_files(
            fixture_path=args.fixture,
            models=_model_list(args.models),
            output_dir=args.output_dir,
            require_reviewed=args.require_reviewed,
        )
    elif args.command == "convert-batch-results":
        report = convert_batch_results(
            mapping_path=args.mapping,
            batch_output_path=args.batch_output,
            model_name=args.model,
            batch_id=args.batch_id,
            batch_status=args.batch_status,
            latency_seconds=args.latency_seconds,
            output_path=args.output,
        )
    else:
        report = compare_results(
            fixture_path=args.fixture,
            result_paths=args.results,
            pricing_path=args.pricing,
            backfill_rows=args.backfill_rows,
            recurring_rows=args.recurring_rows,
            require_reviewed=args.require_reviewed,
        )
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
