"""
instagram_template_pipeline.py

Template-driven Instagram infographic test pipeline.

Primary path:
- build post context from existing S3 datasets
- map context fields to external template placeholders
- render via Bannerbear when configured

Fallback path:
- reuse the existing local HTML/CSS renderer as a mock renderer when
  Bannerbear credentials or template IDs are not available yet

Scope:
- visuals only
- no captions
- no post copy
- explicit template field mapping
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

from process.instagram_render_post import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    S3CSVLoader,
    build_post_context,
    make_issue_rows,
    render_slides,
    screenshot_html_files,
)


BANNERBEAR_API_BASE = "https://api.bannerbear.com/v2"
BANNERBEAR_SYNC_API_BASE = "https://sync.api.bannerbear.com/v2"


class TemplatePipelineError(RuntimeError):
    pass


class BannerbearConfigError(TemplatePipelineError):
    pass


class BannerbearRenderError(TemplatePipelineError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to YAML post spec")
    parser.add_argument("--provider", help="Optional provider override")
    parser.add_argument("--constituency", help="Optional override for constituency name")
    parser.add_argument("--member-name", help="Optional override for member name")
    parser.add_argument("--output-dir", help="Optional override for output root")
    parser.add_argument("--skip-fallback", action="store_true", help="Fail instead of falling back to local HTML rendering")
    parser.add_argument("--skip-screenshots", action="store_true", help="Skip local HTML screenshot export")
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def coalesce_text(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return None


def get_path(payload: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            return default
    return current


def format_issue_summary(counter_payload: Dict[str, int], issue_limit: int = 8) -> str:
    rows, _axis = make_issue_rows(counter=Counter(counter_payload), limit=issue_limit)
    if not rows:
        return "No classified issue counts available yet."
    lines = []
    for idx, row in enumerate(rows, start=1):
        lines.append(f"{idx}. {row['label']} — {row['count']}")
    return "\n".join(lines)


def enrich_context(context: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    issue_limit = int(spec.get("data", {}).get("issue_limit", 8) or 8)
    context["computed"] = {
        "constituency_issue_summary": format_issue_summary(
            context.get("constituency_issue_counts", {}),
            issue_limit=issue_limit,
        ),
        "member_issue_summary": format_issue_summary(
            context.get("member_issue_counts", {}),
            issue_limit=issue_limit,
        ),
        "datasets_used_text": "\n".join(context.get("datasets_used", [])),
        "member_background_short": coalesce_text(context.get("member", {}).get("background"), "Background not available yet."),
    }
    return context


def load_spec(args: argparse.Namespace) -> Dict[str, Any]:
    spec_path = Path(args.spec)
    spec = read_yaml(spec_path)
    if args.constituency:
        spec.setdefault("data", {})["constituency"] = args.constituency
    if args.member_name:
        spec.setdefault("data", {})["member_name"] = args.member_name
    if args.output_dir:
        spec.setdefault("post", {})["output_root"] = args.output_dir
    if args.provider:
        spec.setdefault("render", {})["provider"] = args.provider
    spec.setdefault("render", {})["template_mapping"] = spec.get("render", {}).get("template_mapping")
    return spec


def build_output_dir(spec: Dict[str, Any]) -> Path:
    output_root = Path(spec["post"].get("output_root", "generated_posts"))
    return output_root / spec["post"]["slug"]


def resolve_template_uid(raw_value: str) -> str:
    value = str(raw_value).strip()
    if value.startswith("env:"):
        env_name = value.split(":", 1)[1].strip()
        resolved = os.getenv(env_name, "").strip()
        if not resolved:
            raise BannerbearConfigError(f"Missing required environment variable: {env_name}")
        return resolved
    if not value:
        raise BannerbearConfigError("Template UID mapping is empty.")
    return value


def apply_transform(value: Any, transform: Optional[str]) -> Any:
    if transform in {None, "", "identity"}:
        return value
    if transform == "string":
        return "" if value is None else str(value)
    if transform == "int_string":
        try:
            return str(int(value))
        except Exception:
            return "0"
    if transform == "multiline":
        return "" if value is None else str(value)
    if transform == "default_image_url":
        return coalesce_text(value, "https://placehold.co/1080x1350/png?text=Image+pending")
    raise TemplatePipelineError(f"Unsupported transform: {transform}")


def build_modifications(payload: Dict[str, Any], mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    modifications: List[Dict[str, Any]] = []
    for item in mapping.get("modifications", []):
        placeholder_name = item["name"]
        field_name = item.get("field") or item.get("kind") or "text"
        value = get_path(payload, item["path"], item.get("default"))
        value = apply_transform(value, item.get("transform"))
        if item.get("skip_if_blank") and not coalesce_text(value):
            continue
        modifications.append({"name": placeholder_name, field_name: value})
    return modifications


class BannerbearRenderer:
    def __init__(self, *, api_key: str, use_sync_api: bool, timeout_seconds: int, poll_interval_seconds: int) -> None:
        self.api_key = api_key
        self.use_sync_api = use_sync_api
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.base_url = BANNERBEAR_SYNC_API_BASE if use_sync_api else BANNERBEAR_API_BASE
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def create_image(self, *, template_uid: str, modifications: List[Dict[str, Any]], metadata: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "template": template_uid,
            "modifications": modifications,
        }
        if metadata:
            body["metadata"] = metadata

        response = self.session.post(f"{self.base_url}/images", json=body, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        if self.use_sync_api:
            status = str(data.get("status", "")).lower()
            if status and status != "completed":
                raise BannerbearRenderError(f"Synchronous Bannerbear render did not complete. Status: {status}")
            return data
        return self.poll_image(data["uid"])

    def poll_image(self, uid: str) -> Dict[str, Any]:
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            response = self.session.get(f"{BANNERBEAR_API_BASE}/images/{uid}", timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            status = str(data.get("status", "")).lower()
            if status == "completed":
                return data
            if status == "failed":
                raise BannerbearRenderError(f"Bannerbear render failed for {uid}")
            time.sleep(self.poll_interval_seconds)
        raise BannerbearRenderError(f"Timed out waiting for Bannerbear image {uid}")

    def download_image(self, image_url: str, out_path: Path) -> None:
        response = self.session.get(image_url, timeout=self.timeout_seconds)
        response.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response.content)


def render_with_bannerbear(*, spec: Dict[str, Any], context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    render_cfg = spec.get("render", {})
    bannerbear_cfg = render_cfg.get("bannerbear", {})
    mapping_path = Path(render_cfg["template_mapping"])
    mapping = read_yaml(mapping_path)

    api_key = os.getenv(bannerbear_cfg.get("api_key_env", "BANNERBEAR_API_KEY"), "").strip()
    if not api_key:
        raise BannerbearConfigError("Missing Bannerbear API key environment variable.")

    renderer = BannerbearRenderer(
        api_key=api_key,
        use_sync_api=bool(bannerbear_cfg.get("use_sync_api", True)),
        timeout_seconds=int(bannerbear_cfg.get("timeout_seconds", 45)),
        poll_interval_seconds=int(bannerbear_cfg.get("poll_interval_seconds", 2)),
    )

    requests_dir = output_dir / "bannerbear" / "requests"
    responses_dir = output_dir / "bannerbear" / "responses"
    png_dir = output_dir / "png"
    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    enabled_slides = [slide for slide in spec["slides"] if slide.get("enabled", True)]
    run_summary: List[Dict[str, Any]] = []

    for idx, slide in enumerate(enabled_slides, start=1):
        slide_key = slide["key"]
        slide_mapping = mapping.get("slides", {}).get(slide_key)
        if not slide_mapping:
            raise BannerbearConfigError(f"No Bannerbear mapping found for slide key: {slide_key}")

        payload = {
            **context,
            "slide": slide,
        }
        template_uid = resolve_template_uid(slide_mapping["template_uid"])
        modifications = build_modifications(payload, slide_mapping)
        request_payload = {
            "template": template_uid,
            "slide_key": slide_key,
            "modifications": modifications,
        }
        request_file = requests_dir / f"{idx:02d}_{slide_key}.json"
        write_json(request_file, request_payload)

        response_data = renderer.create_image(
            template_uid=template_uid,
            modifications=modifications,
            metadata=f"{spec['post']['slug']}::{slide_key}",
        )
        response_file = responses_dir / f"{idx:02d}_{slide_key}.json"
        write_json(response_file, response_data)

        image_url = response_data.get("image_url")
        if not image_url:
            raise BannerbearRenderError(f"Bannerbear response missing image_url for slide: {slide_key}")

        image_path = png_dir / f"{idx:02d}_{slide_key}.png"
        renderer.download_image(image_url, image_path)
        run_summary.append(
            {
                "slide_key": slide_key,
                "template_uid": template_uid,
                "image_url": image_url,
                "local_path": str(image_path),
            }
        )

    summary = {
        "provider": "bannerbear",
        "status": "completed",
        "slides": run_summary,
    }
    write_json(output_dir / "bannerbear_run_summary.json", summary)
    return summary


def render_with_local_html(*, spec: Dict[str, Any], context: Dict[str, Any], output_dir: Path, skip_screenshots: bool) -> Dict[str, Any]:
    html_paths = render_slides(spec, context, output_dir)
    if not skip_screenshots:
        asyncio.run(
            screenshot_html_files(
                html_paths=html_paths,
                output_dir=output_dir,
                width=int(spec["post"]["slide_size"]["width"]),
                height=int(spec["post"]["slide_size"]["height"]),
            )
        )
    summary = {
        "provider": "local_html",
        "status": "completed",
        "html_files": [str(path) for path in html_paths],
    }
    write_json(output_dir / "local_html_run_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    spec = load_spec(args)
    output_dir = build_output_dir(spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = S3CSVLoader(bucket=DEFAULT_BUCKET, region=DEFAULT_REGION)
    context = build_post_context(spec, loader)
    context = enrich_context(context, spec)
    write_json(output_dir / "post_context.json", context)

    provider = str(spec.get("render", {}).get("provider", "bannerbear")).strip().lower()
    fallback_provider = str(spec.get("render", {}).get("fallback_provider", "local_html")).strip().lower()

    status_payload: Dict[str, Any] = {
        "requested_provider": provider,
        "fallback_provider": fallback_provider,
        "used_provider": None,
        "fallback_used": False,
        "error": None,
    }

    try:
        if provider == "bannerbear":
            render_with_bannerbear(spec=spec, context=context, output_dir=output_dir)
        elif provider == "local_html":
            render_with_local_html(spec=spec, context=context, output_dir=output_dir, skip_screenshots=args.skip_screenshots)
        else:
            raise TemplatePipelineError(f"Unsupported provider: {provider}")
        status_payload["used_provider"] = provider
    except TemplatePipelineError as exc:
        status_payload["error"] = str(exc)
        if args.skip_fallback or fallback_provider != "local_html" or provider == "local_html":
            write_json(output_dir / "render_status.json", status_payload)
            raise
        render_with_local_html(spec=spec, context=context, output_dir=output_dir, skip_screenshots=args.skip_screenshots)
        status_payload["used_provider"] = "local_html"
        status_payload["fallback_used"] = True

    write_json(output_dir / "render_status.json", status_payload)
    print(json.dumps(status_payload, indent=2))
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
