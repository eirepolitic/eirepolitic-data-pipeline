from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .constants import DEFAULT_BUCKET, DEFAULT_REGION, OUTPUT_ROOT
from .context import build_post_context
from .data_loader import LocalCSVLoader, S3CSVLoader, load_datasets
from .slides import render_slides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render deterministic Instagram infographic PNG slides.")
    parser.add_argument("--spec", required=True, help="Path to the YAML post spec.")
    parser.add_argument("--constituency", help="Optional constituency override.")
    parser.add_argument("--member-name", help="Optional member name override.")
    parser.add_argument("--output-dir", help="Optional output root override.")
    parser.add_argument("--data-source", choices=["s3", "local"], default="s3")
    parser.add_argument("--data-root", help="Required when --data-source=local.")
    parser.add_argument("--s3-bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--aws-region", default=DEFAULT_REGION)
    return parser.parse_args()


def load_spec(path: str | Path) -> Dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "post" not in spec or "slides" not in spec or "data" not in spec or "branding" not in spec:
        raise RuntimeError("Spec must include post, slides, data, and branding sections.")
    return spec


def build_loader(args: argparse.Namespace):
    if args.data_source == "local":
        if not args.data_root:
            raise RuntimeError("--data-root is required when --data-source=local")
        return LocalCSVLoader(args.data_root)
    return S3CSVLoader(bucket=args.s3_bucket, region=args.aws_region)


def main() -> None:
    args = parse_args()
    spec = load_spec(args.spec)

    if args.constituency:
        spec.setdefault("data", {})["constituency"] = args.constituency
    if args.member_name:
        spec.setdefault("data", {})["member_name"] = args.member_name
    if args.output_dir:
        spec.setdefault("post", {})["output_root"] = args.output_dir

    output_root = Path(spec["post"].get("output_root", OUTPUT_ROOT))
    post_slug = spec["post"]["slug"]
    output_dir = output_root / post_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args)
    bundle = load_datasets(loader)
    context = build_post_context(spec, bundle)

    context_path = output_dir / "post_context.json"
    context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")

    paths = render_slides(context, output_dir)
    print(f"Done. Output folder: {output_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
