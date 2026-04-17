from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from openai import OpenAI


DEFAULT_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
DEFAULT_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1536")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--jobs-path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jobs(run_root: Path, jobs_path: str | None) -> List[Dict[str, Any]]:
    path = Path(jobs_path) if jobs_path else run_root / "jobs" / "generation_jobs.jsonl"
    jobs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                jobs.append(json.loads(text))
    return jobs


def build_render_spec(job: Dict[str, Any], image_path: Path, output_root: Path) -> Dict[str, Any]:
    base_spec_path = Path(job["base_spec_path"])
    spec = yaml.safe_load(base_spec_path.read_text(encoding="utf-8"))
    spec["post"]["slug"] = job["record_id"]
    spec["post"]["output_root"] = str(output_root.resolve())
    spec.setdefault("data", {})["constituency"] = job["constituency_name"]
    spec["data"]["generated_background_image_path"] = image_path.resolve().as_uri()
    spec.setdefault("branding", {})["footer_note"] = (
        f"Option 5 test • {job['style_direction']} • v{int(job['variant_index']):02d}"
    )
    return spec


def blank_review_row(manifest_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **manifest_row,
        "brand_consistency": "",
        "factual_correctness_visible_text": "",
        "text_legibility": "",
        "repeatability_note": "",
        "better_than_deterministic_template": "",
        "approved": "",
        "review_notes": "",
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    jobs = load_jobs(run_root, args.jobs_path)

    client = OpenAI()

    image_dir = run_root / "images"
    metadata_dir = run_root / "metadata"
    render_spec_dir = run_root / "render_specs"
    render_output_root = run_root / "rendered_posts"
    review_dir = run_root / "review"

    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    render_spec_dir.mkdir(parents=True, exist_ok=True)
    render_output_root.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []

    for job in jobs:
        image_path = image_dir / job["image_filename"]
        metadata_path = metadata_dir / f"{Path(job['image_filename']).stem}.json"
        render_spec_path = render_spec_dir / f"{job['record_id']}.yml"

        if image_path.exists() and not args.overwrite:
            status = "reused_existing"
        else:
            result = client.images.generate(
                model=args.model,
                prompt=job["prompt"],
                size=args.size,
            )
            b64_json = result.data[0].b64_json
            if not b64_json:
                raise RuntimeError(f"No image payload returned for {job['record_id']}")
            image_bytes = base64.b64decode(b64_json)
            image_path.write_bytes(image_bytes)
            status = "generated"

            response_dump = result.model_dump() if hasattr(result, "model_dump") else {"raw_result": str(result)}
            metadata = {
                "record_id": job["record_id"],
                "status": status,
                "model": args.model,
                "size": args.size,
                "job": job,
                "response": response_dump,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        render_spec = build_render_spec(job, image_path, render_output_root)
        render_spec_path.write_text(
            yaml.safe_dump(render_spec, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        manifest_row = {
            "record_id": job["record_id"],
            "constituency_name": job["constituency_name"],
            "style_direction": job["style_direction"],
            "variant_index": job["variant_index"],
            "prompt_hash": job["prompt_hash"],
            "model": args.model,
            "size": args.size,
            "image_path": str(image_path.resolve()),
            "render_spec_path": str(render_spec_path.resolve()),
            "status": status,
        }
        manifest_rows.append(manifest_row)
        review_rows.append(blank_review_row(manifest_row))

    manifest_jsonl_path = run_root / "metadata" / "generated_manifest.jsonl"
    with manifest_jsonl_path.open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_csv_path = run_root / "metadata" / "generated_manifest.csv"
    with manifest_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)

    review_csv_path = review_dir / "review_sheet.csv"
    with review_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(review_rows[0].keys()) if review_rows else [])
        if review_rows:
            writer.writeheader()
            writer.writerows(review_rows)

    print(f"Generated or reused {len(manifest_rows)} images.")
    print(f"Manifest CSV: {manifest_csv_path.resolve()}")
    print(f"Review CSV: {review_csv_path.resolve()}")
    print("Render spec files:")
    for row in manifest_rows:
        print(f"- {row['render_spec_path']}")


if __name__ == "__main__":
    main()
