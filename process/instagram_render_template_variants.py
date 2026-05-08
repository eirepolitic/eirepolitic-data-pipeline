from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from instagram.renderer.template_renderer import render_template_file
from process.instagram_render_campaign import read_csv, row_bindings, select_rows, slugify, value, write_review


def template_id_from_path(path: str | Path) -> str:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(doc.get("template_id") or Path(path).stem)


def render_variants(spec_path: str | Path, limit_override: int | None = None) -> dict[str, Any]:
    spec_path = Path(spec_path)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if spec.get("campaign") != "member_profile_batch_v1":
        raise RuntimeError("Only member_profile_batch_v1 is supported by this variant renderer.")

    render = spec.get("render", {})
    templates = render.get("variant_templates") or [render.get("template")]
    templates = [str(t) for t in templates if t]
    if not templates:
        raise RuntimeError("No render.template or render.variant_templates provided.")

    df = read_csv(spec["data"]["source_table"])
    selected = select_rows(df, spec["variation"]["selector"], limit_override)
    if selected.empty:
        raise RuntimeError("No rows selected for rendering.")

    output_root = Path(render.get("output_root", "generated_posts/member_profile_batch_v1_fixture"))
    png_dir = output_root / "png"
    bindings_dir = output_root / "metadata" / "bindings"
    png_dir.mkdir(parents=True, exist_ok=True)
    bindings_dir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        member_slug = slugify(row.get("full_name"))
        bindings = row_bindings(row, spec.get("review", {}).get("footer_text", "Source: Oireachtas data pipeline"))
        for template_path in templates:
            template_id = template_id_from_path(template_path)
            variant_slug = slugify(template_id)
            slug = f"{variant_slug}-{member_slug}"
            bindings_path = bindings_dir / f"{slug}.yml"
            bindings_path.write_text(yaml.safe_dump({"bindings": bindings}, sort_keys=False, allow_unicode=True), encoding="utf-8")
            out_path = png_dir / f"{slug}.png"
            manifest = render_template_file(template_path, bindings_path, out_path, render.get("palette"))
            review_rows.append({
                "output_file": str(out_path),
                "bindings_file": str(bindings_path),
                "render_manifest_file": str(output_root / "metadata" / "manifests" / f"{slug}.render_manifest.json"),
                "template_id": template_id,
                "full_name": bindings["member_name"],
                "party": bindings["party"],
                "constituency": bindings["constituency"],
                "top_issue_2025": bindings["top_issue"],
                "vote_participation_pct_2025": bindings["vote_participation"],
                "speech_count_2025": value(row, "speech_count_2025", "0"),
                "speech_rank_2025": bindings["speech_rank"],
                "photo_url": bindings["member_photo"],
                "warnings": ";".join(manifest.get("warnings", [])),
            })

    write_review(output_root, review_rows, spec, spec_path)
    return {"success": True, "campaign": spec["campaign"], "rendered": len(review_rows), "templates": templates, "output_root": str(output_root)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one campaign fixture through one or more JSON templates.")
    parser.add_argument("--campaign", required=True, help="Path to campaign render spec YAML")
    parser.add_argument("--limit", type=int, help="Optional row limit override")
    args = parser.parse_args()
    print(json.dumps(render_variants(args.campaign, args.limit), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
