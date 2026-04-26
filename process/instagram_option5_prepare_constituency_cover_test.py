from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from instagram_render_post import S3CSVLoader, normalize_constituency, pick_constituency_image


DEFAULT_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
DEFAULT_REGION = os.getenv("AWS_REGION", "ca-central-1")


def slugify(value: str) -> str:
    return normalize_constituency(value).replace(" ", "-")


def write_github_output(key: str, value: str) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(f"{key}={value}\n")


def build_prompt(
    *,
    constituency_name: str,
    reference_image_url: str | None,
    style_direction: str,
    style_fragment: str,
    safe_text_policy: str,
    prompt_cfg: Dict[str, Any],
) -> str:
    composition_rules = prompt_cfg.get("composition_rules", [])
    negative_rules = prompt_cfg.get("negative_rules", [])
    canvas_goal = prompt_cfg.get("canvas_goal", "Instagram constituency cover visual")

    lines: List[str] = [
        f"Create a polished portrait 4:5 political infographic cover background for {canvas_goal}.",
        f"Primary subject: the Irish constituency '{constituency_name}'.",
        f"Style direction: {style_direction}. {style_fragment}",
        "This is decorative background artwork only, not a finished post.",
        "Preserve a large clean central title zone for later deterministic text overlay.",
        "The image should feel brand-consistent, premium, restrained, and suitable for an Irish politics explainer page.",
        "Use strong large-scale composition, crisp shape language, and high contrast.",
        "Do not rely on accurate factual micro-details in the artwork.",
        f"Text policy: {safe_text_policy}.",
    ]

    if reference_image_url:
        lines.append(
            f"Reference asset available for inspiration only: {reference_image_url}. "
            "Use it loosely as a shape or silhouette cue, not as a photo to reproduce exactly."
        )

    if composition_rules:
        lines.append("Composition requirements:")
        lines.extend(f"- {rule}" for rule in composition_rules)

    if negative_rules:
        lines.append("Avoid:")
        lines.extend(f"- {rule}" for rule in negative_rules)

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--constituency", required=True)
    parser.add_argument("--variant-count", type=int, default=2)
    parser.add_argument("--style-mode", choices=["both", "map_poster", "textured_editorial"], default="both")
    parser.add_argument("--output-root", default="generated_visual_tests/option5_constituency_cover")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    loader = S3CSVLoader(bucket=DEFAULT_BUCKET, region=DEFAULT_REGION)
    df_images = loader.read_first_csv("constituency_images", ["processed/constituencies/constituency_images.csv"])

    constituency_name = args.constituency.strip()
    reference_image_url = pick_constituency_image(df_images, constituency_name)
    constituency_slug = slugify(constituency_name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(args.output_root) / f"{constituency_slug}__{run_id}"

    (run_root / "inputs").mkdir(parents=True, exist_ok=True)
    (run_root / "jobs").mkdir(parents=True, exist_ok=True)
    (run_root / "images").mkdir(parents=True, exist_ok=True)
    (run_root / "metadata").mkdir(parents=True, exist_ok=True)
    (run_root / "render_specs").mkdir(parents=True, exist_ok=True)
    (run_root / "review").mkdir(parents=True, exist_ok=True)

    style_fragments = spec.get("option5_prompt", {}).get("style_fragments", {})
    if args.style_mode == "both":
        style_directions = ["map_poster", "textured_editorial"]
    else:
        style_directions = [args.style_mode]

    jobs: List[Dict[str, Any]] = []
    truth_text = {
        "slide_title": spec.get("data", {}).get("cover_label") or "Constituency Profile",
        "constituency_name": constituency_name,
        "footer_note": spec.get("branding", {}).get("footer_note"),
    }

    source_snapshot = {
        "constituency_name": constituency_name,
        "constituency_slug": constituency_slug,
        "reference_image_url": reference_image_url,
        "datasets_used": {
            "constituency_images": loader.used_keys.get("constituency_images"),
        },
        "visible_truth_text": truth_text,
        "risk_notes": [
            "Generated visuals are not trusted for factual correctness by appearance alone.",
            "Exact visible title text is overlaid deterministically during render review.",
            "Small text and numeric claims are intentionally excluded from this first experiment.",
        ],
    }

    base_spec = deepcopy(spec)
    base_spec.setdefault("data", {})["constituency"] = constituency_name
    (run_root / "inputs" / "source_snapshot.json").write_text(
        json.dumps(source_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_root / "inputs" / "base_render_spec.yml").write_text(
        yaml.safe_dump(base_spec, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    for style_direction in style_directions:
        style_fragment = style_fragments.get(style_direction, style_direction)
        for variant_index in range(1, args.variant_count + 1):
            prompt = build_prompt(
                constituency_name=constituency_name,
                reference_image_url=reference_image_url,
                style_direction=style_direction,
                style_fragment=style_fragment,
                safe_text_policy=spec.get("experiment", {}).get("safe_text_policy", "deterministic_overlay_only"),
                prompt_cfg=spec.get("option5_prompt", {}),
            )
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            record_id = f"{constituency_slug}__{style_direction}__v{variant_index:02d}"
            jobs.append(
                {
                    "record_id": record_id,
                    "constituency_name": constituency_name,
                    "constituency_slug": constituency_slug,
                    "style_direction": style_direction,
                    "variant_index": variant_index,
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "reference_image_url": reference_image_url,
                    "visible_truth_text": truth_text,
                    "base_spec_path": str((run_root / "inputs" / "base_render_spec.yml").resolve()),
                    "run_root": str(run_root.resolve()),
                    "image_filename": f"{record_id}__{prompt_hash}.png",
                }
            )

    jobs_path = run_root / "jobs" / "generation_jobs.jsonl"
    with jobs_path.open("w", encoding="utf-8") as fh:
        for job in jobs:
            fh.write(json.dumps(job, ensure_ascii=False) + "\n")

    (run_root / "jobs" / "generation_jobs.pretty.json").write_text(
        json.dumps(jobs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_github_output("run_root", str(run_root.resolve()))
    write_github_output("jobs_path", str(jobs_path.resolve()))

    print(f"Prepared {len(jobs)} jobs.")
    print(f"Run root: {run_root.resolve()}")
    print(f"Jobs: {jobs_path.resolve()}")


if __name__ == "__main__":
    main()
