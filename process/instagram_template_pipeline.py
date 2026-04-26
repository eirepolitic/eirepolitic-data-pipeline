"""
instagram_template_pipeline.py

Template-driven Instagram infographic test pipeline.

Primary paths:
- build post context from existing S3 datasets
- map context fields to external template placeholders
- render via Bannerbear or Placid when configured

Fallback path:
- reuse the existing local HTML/CSS renderer as a mock renderer when
  external provider credentials or template IDs are not available yet

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

try:
    from process.instagram_render_post import (
        DEFAULT_BUCKET,
        DEFAULT_REGION,
        S3CSVLoader,
        build_post_context,
        make_issue_rows,
        render_slides,
        screenshot_html_files,
    )
except ModuleNotFoundError:
    from instagram_render_post import (
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
PLACID_API_BASE = "https://api.placid.app/api/rest"


class TemplatePipelineError(RuntimeError):
    pass


class ProviderConfigError(TemplatePipelineError):
    pass


class ProviderRenderError(TemplatePipelineError):
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
        "member_background_short": coalesce_text(
            context.get("member", {}).get("background"),
            "Background not available yet.",
        ),
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
    return spec


def build_output_dir(spec: Dict[str, Any]) -> Path:
    output_root = Path(spec["post"].get("output_root", "generated_posts"))
    return output_root / spec["post"]["slug"]


def resolve_reference(raw_value: str) -> str:
    value = str(raw_value).strip()
    if value.startswith("env:"):
        env_name = value.split(":", 1)[1].strip()
        resolved = os.getenv(env_name, "").strip()
        if not resolved:
            raise ProviderConfigError(f"Missing required environment variable: {env_name}")
        return resolved
    if not value:
        raise ProviderConfigError("Template reference is empty.")
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


def build_bannerbear_modifications(payload: Dict[str, Any], mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def build_placid_layers(payload: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    layers: Dict[str, Dict[str, Any]] = {}
    for item in mapping.get("layers", []):
        layer_name = item["name"]
        property_name = item.get("property") or item.get("kind") or "text"
        value = get_path(payload, item["path"], item.get("default"))
        value = apply_transform(value, item.get("transform"))
        if item.get("skip_if_blank") and not coalesce_text(value):
            continue
        layers[layer_name] = {property_name: value}
    return layers


class BaseAPIClient:
    def __init__(self, *, api_key: str) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def download_binary(self, url: str, out_path: Path, timeout_seconds: int) -> None:
        response = self.session.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response.content)


class BannerbearRenderer(BaseAPIClient):
    def __init__(self, *, api_key: str, use_sync_api: bool, timeout_seconds: int, poll_interval_seconds: int) -> None:
        super().__init__(api_key=api_key)
        self.use_sync_api = use_sync_api
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.base_url = BANNERBEAR_SYNC_API_BASE if use_sync_api else BANNERBEAR_API_BASE

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
                raise ProviderRenderError(f"Synchronous Bannerbear render did not complete. Status: {status}")
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
                raise ProviderRenderError(f"Bannerbear render failed for {uid}")
            time.sleep(self.poll_interval_seconds)
        raise ProviderRenderError(f"Timed out waiting for Bannerbear image {uid}")


class PlacidRenderer(BaseAPIClient):
    def __init__(self, *, api_key: str, create_now: bool, timeout_seconds: int, poll_interval_seconds: int) -> None:
        super().__init__(api_key=api_key)
        self.create_now = create_now
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    def create_image(
        self,
        *,
        template_uuid: str,
        layers: Dict[str, Dict[str, Any]],
        metadata: Optional[str],
        width: int,
        height: int,
        filename: str,
        image_format: str,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "template_uuid": template_uuid,
            "create_now": self.create_now,
            "layers": layers,
            "modifications": {
                "width": width,
                "height": height,
                "filename": filename,
                "image_format": image_format,
            },
        }
        if metadata:
            body["passthrough"] = metadata

        response = self.session.post(f"{PLACID_API_BASE}/images", json=body, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()

        status = str(data.get("status", "")).lower()
        if status == "finished" and data.get("image_url"):
            return data

        polling_url = data.get("polling_url")
        if not polling_url:
            raise ProviderRenderError("Placid response missing polling_url before image finished.")
        return self.poll_image(polling_url)

    def poll_image(self, polling_url: str) -> Dict[str, Any]:
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            response = self.session.get(polling_url, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            status = str(data.get("status", "")).lower()
            if status == "finished" and data.get("image_url"):
                return data
            if status == "error":
                raise ProviderRenderError(f"Placid render failed. Response: {data}")
            time.sleep(self.poll_interval_seconds)
        raise ProviderRenderError("Timed out waiting for Placid image")


def render_with_bannerbear(*, spec: Dict[str, Any], context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    render_cfg = spec.get("render", {})
    bannerbear_cfg = render_cfg.get("bannerbear", {})
    mapping = read_yaml(Path(render_cfg["template_mapping"]))

    api_key = os.getenv(bannerbear_cfg.get("api_key_env", "BANNERBEAR_API_KEY"), "").strip()
    if not api_key:
        raise ProviderConfigError("Missing Bannerbear API key environment variable.")

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
            raise ProviderConfigError(f"No Bannerbear mapping found for slide key: {slide_key}")

        payload = {**context, "slide": slide}
        template_uid = resolve_reference(slide_mapping["template_uid"])
        modifications = build_bannerbear_modifications(payload, slide_mapping)
        request_payload = {
            "provider": "bannerbear",
            "template": template_uid,
            "slide_key": slide_key,
            "modifications": modifications,
        }
        write_json(requests_dir / f"{idx:02d}_{slide_key}.json", request_payload)

        response_data = renderer.create_image(
            template_uid=template_uid,
            modifications=modifications,
            metadata=f"{spec['post']['slug']}::{slide_key}",
        )
        write_json(responses_dir / f"{idx:02d}_{slide_key}.json", response_data)

        image_url = response_data.get("image_url")
        if not image_url:
            raise ProviderRenderError(f"Bannerbear response missing image_url for slide: {slide_key}")

        image_path = png_dir / f"{idx:02d}_{slide_key}.png"
        renderer.download_binary(image_url, image_path, renderer.timeout_seconds)
        run_summary.append(
            {
                "slide_key": slide_key,
                "template_uid": template_uid,
                "image_url": image_url,
                "local_path": str(image_path),
            }
        )

    summary = {"provider": "bannerbear", "status": "completed", "slides": run_summary}
    write_json(output_dir / "bannerbear_run_summary.json", summary)
    return summary


def render_with_placid(*, spec: Dict[str, Any], context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    render_cfg = spec.get("render", {})
    placid_cfg = render_cfg.get("placid", {})
    mapping = read_yaml(Path(render_cfg["template_mapping"]))

    api_key = os.getenv(placid_cfg.get("api_key_env", "PLACID_API_TOKEN"), "").strip()
    if not api_key:
        raise ProviderConfigError("Missing Placid API token environment variable.")

    renderer = PlacidRenderer(
        api_key=api_key,
        create_now=bool(placid_cfg.get("create_now", True)),
        timeout_seconds=int(placid_cfg.get("timeout_seconds", 60)),
        poll_interval_seconds=int(placid_cfg.get("poll_interval_seconds", 2)),
    )

    requests_dir = output_dir / "placid" / "requests"
    responses_dir = output_dir / "placid" / "responses"
    png_dir = output_dir / "png"
    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    width = int(spec["post"]["slide_size"]["width"])
    height = int(spec["post"]["slide_size"]["height"])
    image_format = str(placid_cfg.get("image_format", "png"))

    enabled_slides = [slide for slide in spec["slides"] if slide.get("enabled", True)]
    run_summary: List[Dict[str, Any]] = []

    for idx, slide in enumerate(enabled_slides, start=1):
        slide_key = slide["key"]
        slide_mapping = mapping.get("slides", {}).get(slide_key)
        if not slide_mapping:
            raise ProviderConfigError(f"No Placid mapping found for slide key: {slide_key}")

        payload = {**context, "slide": slide}
        template_uuid = resolve_reference(slide_mapping["template_uuid"])
        layers = build_placid_layers(payload, slide_mapping)
        filename = f"{idx:02d}_{slide_key}.{image_format}"

        request_payload = {
            "provider": "placid",
            "template_uuid": template_uuid,
            "slide_key": slide_key,
            "layers": layers,
            "modifications": {
                "width": width,
                "height": height,
                "filename": filename,
                "image_format": image_format,
            },
        }
        write_json(requests_dir / f"{idx:02d}_{slide_key}.json", request_payload)

        response_data = renderer.create_image(
            template_uuid=template_uuid,
            layers=layers,
            metadata=f"{spec['post']['slug']}::{slide_key}",
            width=width,
            height=height,
            filename=filename,
            image_format=image_format,
        )
        write_json(responses_dir / f"{idx:02d}_{slide_key}.json", response_data)

        image_url = response_data.get("image_url")
        if not image_url:
            raise ProviderRenderError(f"Placid response missing image_url for slide: {slide_key}")

        image_path = png_dir / f"{idx:02d}_{slide_key}.png"
        renderer.download_binary(image_url, image_path, renderer.timeout_seconds)
        run_summary.append(
            {
                "slide_key": slide_key,
                "template_uuid": template_uuid,
                "image_url": image_url,
                "local_path": str(image_path),
            }
        )

    summary = {"provider": "placid", "status": "completed", "slides": run_summary}
    write_json(output_dir / "placid_run_summary.json", summary)
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
    context = enrich_context(build_post_context(spec, loader), spec)
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
        elif provider == "placid":
            render_with_placid(spec=spec, context=context, output_dir=output_dir)
        elif provider == "local_html":
            render_with_local_html(
                spec=spec,
                context=context,
                output_dir=output_dir,
                skip_screenshots=args.skip_screenshots,
            )
        else:
            raise TemplatePipelineError(f"Unsupported provider: {provider}")
        status_payload["used_provider"] = provider
    except TemplatePipelineError as exc:
        status_payload["error"] = str(exc)
        if args.skip_fallback or fallback_provider != "local_html" or provider == "local_html":
            write_json(output_dir / "render_status.json", status_payload)
            raise
        render_with_local_html(
            spec=spec,
            context=context,
            output_dir=output_dir,
            skip_screenshots=args.skip_screenshots,
        )
        status_payload["used_provider"] = "local_html"
        status_payload["fallback_used"] = True

    write_json(output_dir / "render_status.json", status_payload)
    print(json.dumps(status_payload, indent=2))
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
