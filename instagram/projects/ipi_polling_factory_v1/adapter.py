from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from PIL import Image

from instagram.factory.package import deterministic_zip
from instagram.factory.render_primitives import contact_sheet
from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers import horizontal_bar
from instagram.projects.ipi_polling_factory_v1.renderers import render_diverging, render_methodology, render_trend

PROJECT_ID = "ipi_polling_factory_v1"
RAW_PARTIES = [
    ("SF", "Sinn Féin"),
    ("FG", "Fine Gael"),
    ("FF", "Fianna Fáil"),
    ("LAB", "Labour"),
    ("SD", "Social Democrats"),
    ("GP", "Green Party"),
    ("SPBP", "PBP-Solidarity"),
    ("AU", "Aontú"),
    ("II", "Independent Ireland"),
    ("IND_OTH_IT", "Independents / Other"),
]
REQUIRED_POLL_COLUMNS = ("date", "date_start", "date_end", "pollster", "sample_size")


def _read_csv(uri: str) -> pd.DataFrame:
    if not uri.startswith("s3://"):
        return pd.read_csv(uri)
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket or not key:
        raise RuntimeError(f"Invalid S3 URI: {uri}")
    s3 = boto3.client("s3", region_name="ca-central-1")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _prepare_polls(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("IPI raw polls table is empty")
    missing = [column for column in REQUIRED_POLL_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"IPI raw polls table is missing required columns: {missing}")
    if not any(code in df.columns for code, _ in RAW_PARTIES):
        raise RuntimeError("IPI raw polls table contains no recognised party columns")

    result = df.copy()
    for source_col, parsed_col in (("date", "_date"), ("date_start", "_date_start"), ("date_end", "_date_end")):
        result[parsed_col] = pd.to_datetime(result[source_col], format="%Y-%m-%d", errors="coerce")
        if result[parsed_col].isna().any():
            raise RuntimeError(f"IPI raw polls table contains invalid {source_col} values")
    result["_sample_size"] = pd.to_numeric(result["sample_size"], errors="coerce")
    if result["_sample_size"].isna().any() or (result["_sample_size"] <= 0).any():
        raise RuntimeError("IPI raw polls table contains invalid sample_size values")
    result["pollster"] = result["pollster"].astype(str).str.strip()
    return result


def _poll_metadata(row: pd.Series) -> dict[str, Any]:
    return {
        "publication_date": str(row["date"]),
        "fieldwork_start": str(row["date_start"]),
        "fieldwork_end": str(row["date_end"]),
        "pollster": str(row["pollster"]),
        "sample_size": int(float(row["_sample_size"])),
        "quality_flags": str(row.get("quality_flags") or "").strip(),
    }


def _latest_poll(polls: pd.DataFrame) -> pd.Series:
    latest_date = polls["_date"].max()
    rows = polls.loc[polls["_date"].eq(latest_date)].copy()
    sort_columns = [column for column in ("source_row_number", "pollster") if column in rows.columns]
    if sort_columns:
        rows = rows.sort_values(sort_columns)
    return rows.iloc[-1]


def _select_previous_same_pollster(
    polls: pd.DataFrame,
    latest: pd.Series,
    *,
    target_days: int,
    minimum_days: int,
    maximum_days: int,
) -> tuple[pd.Series, int, str]:
    pollster = str(latest["pollster"])
    latest_date = latest["_date"]
    earlier = polls.loc[polls["pollster"].eq(pollster) & polls["_date"].lt(latest_date)].copy()
    if earlier.empty:
        raise RuntimeError(f"No previous poll is available from {pollster}")
    earlier["_days_before"] = (latest_date - earlier["_date"]).dt.days
    preferred = earlier.loc[earlier["_days_before"].between(minimum_days, maximum_days, inclusive="both")].copy()
    if not preferred.empty:
        preferred["_distance_from_target"] = (preferred["_days_before"] - target_days).abs()
        preferred = preferred.sort_values(["_distance_from_target", "_date"], ascending=[True, False])
        row = preferred.iloc[0]
        return row, int(row["_days_before"]), "target_window"
    earlier = earlier.sort_values("_date", ascending=False)
    row = earlier.iloc[0]
    return row, int(row["_days_before"]), "fallback_previous_wave"


def _numeric(row: pd.Series, key: str) -> float | None:
    if key not in row.index:
        return None
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _latest_rows(latest: pd.Series, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code, label in RAW_PARTIES:
        value = _numeric(latest, code)
        if value is None:
            continue
        rows.append({"party_code": code, "label": label, "value": round(value, 1)})
    return sorted(rows, key=lambda item: item["value"], reverse=True)[:limit]


def _change_rows(latest: pd.Series, previous: pd.Series, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code, label in RAW_PARTIES:
        current = _numeric(latest, code)
        prior = _numeric(previous, code)
        if current is None or prior is None:
            continue
        rows.append(
            {
                "party_code": code,
                "label": label,
                "value": round(current - prior, 1),
                "current_pct": round(current, 1),
                "previous_pct": round(prior, 1),
            }
        )
    return sorted(rows, key=lambda item: item["value"], reverse=True)[:limit]


def _same_pollster_trend(
    polls: pd.DataFrame,
    latest: pd.Series,
    *,
    window_days: int,
    party_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    pollster = str(latest["pollster"])
    end_date = latest["_date"]
    cutoff_date = end_date - pd.Timedelta(days=window_days)
    window = polls.loc[
        polls["pollster"].eq(pollster)
        & polls["_date"].between(cutoff_date, end_date, inclusive="both")
    ].sort_values("_date")
    if len(window) < 2:
        raise RuntimeError(f"Need at least two {pollster} polls within {window_days} days to draw the raw-poll trend")

    ranked: list[tuple[str, str, float]] = []
    for code, label in RAW_PARTIES:
        value = _numeric(latest, code)
        if value is not None:
            ranked.append((code, label, value))
    selected = sorted(ranked, key=lambda item: item[2], reverse=True)[:party_limit]

    series: list[dict[str, Any]] = []
    for code, label, _ in selected:
        points = []
        for _, row in window.iterrows():
            value = _numeric(row, code)
            if value is not None:
                points.append({"date": str(row["date"]), "value": round(value, 1)})
        if len(points) >= 2:
            series.append({"party_code": code, "label": label, "points": points})

    waves = [_poll_metadata(row) for _, row in window.iterrows()]
    if not series:
        raise RuntimeError(f"No comparable party series are available across the selected {pollster} polls")
    actual_span_days = int((window["_date"].max() - window["_date"].min()).days)
    return series, waves, actual_span_days


def _visual_template(project: dict[str, Any], *, value_format: str = "percent") -> dict[str, Any]:
    render_cfg = project.get("render") or {}
    palette = render_cfg.get("palette") or {}
    return {
        "template_id": "horizontal_bar_draft_v1",
        "params": {
            "width": 1032,
            "height": 1210,
            "max_items": int(render_cfg.get("max_items", 8)),
            "sort": "descending",
            "value_format": value_format,
            "min_visual_rows": 4,
            "legend_variant": str(render_cfg.get("trend_legend_variant") or "single_row"),
        },
        "palette": {
            "background": str(palette.get("background") or "#0f2f24"),
            "panel": str(palette.get("background") or "#0f2f24"),
            "text": str(palette.get("text") or "#f4ead7"),
            "muted": str(palette.get("muted") or "#c8bda8"),
            "accent": str(palette.get("accent") or "#d8b45f"),
            "grid": str(palette.get("grid") or "#f4ead7"),
        },
        "series_colors": [str(value) for value in (render_cfg.get("trend_colors") or [])],
    }


def _render_outer(project: dict[str, Any], *, title: str, visual_path: Path, output_path: Path) -> dict[str, Any]:
    layout_path = Path(str((project.get("render") or {})["outer_layout"]))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    result = render_template(layout, {"slide_title": title, "main_media": str(visual_path)}, output_path)
    if result.warnings:
        raise RuntimeError(f"Approved outer layout warnings for {title}: {result.warnings}")
    return {"outer_layout": str(layout_path), "text_metrics": result.text_metrics}


def _assert_slide(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Slide was not created: {path}")
    with Image.open(path) as image:
        if image.size != (1080, 1350):
            raise RuntimeError(f"Unexpected slide dimensions for {path}: {image.size}")


def _pretty_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%-d %b %Y")


def _methodology_entries(
    latest: dict[str, Any],
    previous: dict[str, Any],
    comparison_days: int,
    trend_waves: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    trend_start = trend_waves[0]["publication_date"]
    trend_end = trend_waves[-1]["publication_date"]
    return [
        (
            "What these numbers are",
            "Slides 1–3 use actual published voting-intention polls from one polling company only. No IPI model, smoothing or daily estimate is used.",
        ),
        (
            "Latest poll",
            f"{latest['pollster']}, published {_pretty_date(latest['publication_date'])}; fieldwork {_pretty_date(latest['fieldwork_start'])} to {_pretty_date(latest['fieldwork_end'])}; sample size {latest['sample_size']:,}.",
        ),
        (
            "The comparison",
            f"Slide 2 compares that poll with the same pollster's {_pretty_date(previous['publication_date'])} wave, {comparison_days} days earlier.",
        ),
        (
            "Six-month trend",
            f"Slide 3 shows {len(trend_waves)} same-pollster polls from {_pretty_date(trend_start)} to {_pretty_date(trend_end)}. Every marker is an actual published poll; lines only connect observations.",
        ),
        (
            "Source and methodology",
            "The IPI feed provides pollster, fieldwork dates, publication date, sample size and party results. Full sampling and weighting methodology should be checked in the original pollster release.",
        ),
    ]


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    source_cfg = project.get("source") or {}
    render_cfg = project.get("render") or {}
    polls_uri = str(source_cfg["polls_csv"])
    source_label = str(source_cfg.get("source_label") or "Irish Polling Indicator (IPI) raw polls")
    source_url = str(source_cfg.get("source_url") or "")

    polls = _prepare_polls(_read_csv(polls_uri))
    latest = _latest_poll(polls)
    previous, comparison_days, comparison_selection = _select_previous_same_pollster(
        polls,
        latest,
        target_days=int(render_cfg.get("target_comparison_days", 30)),
        minimum_days=int(render_cfg.get("minimum_comparison_days", 28)),
        maximum_days=int(render_cfg.get("maximum_comparison_days", 45)),
    )
    latest_meta = _poll_metadata(latest)
    previous_meta = _poll_metadata(previous)
    pollster = latest_meta["pollster"]
    limit = int(render_cfg.get("max_items", 8))
    party_limit = int(render_cfg.get("trend_party_limit", 5))
    trend_window_days = int(render_cfg.get("trend_window_days", 183))

    latest_rows = _latest_rows(latest, limit)
    change_rows = _change_rows(latest, previous, limit)
    trend_series, trend_waves, trend_actual_span_days = _same_pollster_trend(
        polls,
        latest,
        window_days=trend_window_days,
        party_limit=party_limit,
    )
    if not latest_rows or not change_rows or not trend_series:
        raise RuntimeError("Raw-poll carousel requires latest, change and trend data")

    latest_date = latest_meta["publication_date"]
    previous_date = previous_meta["publication_date"]
    trend_start_date = trend_waves[0]["publication_date"]
    trend_end_date = trend_waves[-1]["publication_date"]
    period_root = output_root / f"period={latest_date}"
    if period_root.exists():
        shutil.rmtree(period_root)
    slides_dir = period_root / "slides"
    assets_dir = period_root / "assets"
    metadata_dir = period_root / "metadata"
    contact_dir = period_root / "contact_sheets"
    for directory in (slides_dir, assets_dir, metadata_dir, contact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    slide_defs = {str(item["id"]): item for item in (project.get("slides") or {}).get("definitions", [])}
    slide_paths = [
        slides_dir / "01_latest_poll.png",
        slides_dir / "02_change_same_pollster.png",
        slides_dir / "03_six_month_polling.png",
        slides_dir / "04_about_polling.png",
    ]
    visual_paths = [
        assets_dir / "01_latest_poll_visual.png",
        assets_dir / "02_change_visual.png",
        assets_dir / "03_poll_trend_visual.png",
    ]

    latest_manifest = horizontal_bar.render(
        _visual_template(project),
        {
            "visual_id": "ipi-latest-raw-poll",
            "bindings": {"label": "label", "value": "value"},
            "source_note": f"{pollster} · published {_pretty_date(latest_date)} · n={latest_meta['sample_size']:,} · source: IPI raw polls",
            "empty_message": "No current poll results available",
        },
        latest_rows,
        visual_paths[0],
        metadata_dir / "01_latest_poll_visual.json",
        metadata_dir / "01_latest_poll_visual_manifest.json",
        {"project_id": PROJECT_ID, "source_uri": polls_uri, "latest_poll": latest_meta},
    )
    if latest_manifest.get("warnings"):
        raise RuntimeError(f"Approved horizontal-bar renderer warnings: {latest_manifest['warnings']}")

    comparison_label = f"{pollster}: {_pretty_date(previous_date)} → {_pretty_date(latest_date)} ({comparison_days} days)"
    change_manifest = render_diverging(
        _visual_template(project),
        {
            "visual_id": "ipi-raw-poll-change",
            "comparison_label": comparison_label,
            "source_note": "Same pollster comparison · change in displayed poll percentages · source: IPI raw polls",
            "empty_message": "No comparable poll change available",
        },
        change_rows,
        visual_paths[1],
        metadata_dir / "02_change_visual.json",
        metadata_dir / "02_change_visual_manifest.json",
        {
            "project_id": PROJECT_ID,
            "source_uri": polls_uri,
            "latest_poll": latest_meta,
            "previous_poll": previous_meta,
            "comparison_days": comparison_days,
            "comparison_selection": comparison_selection,
        },
    )
    if change_manifest.get("warnings"):
        raise RuntimeError(f"Diverging renderer warnings: {change_manifest['warnings']}")

    range_label = f"{pollster} · {_pretty_date(trend_start_date)} to {_pretty_date(trend_end_date)} · each marker = one actual poll"
    trend_manifest = render_trend(
        _visual_template(project),
        {
            "visual_id": "ipi-raw-poll-trend",
            "source_note": f"{len(trend_waves)} {pollster} polls · source: IPI raw polls",
            "range_label": range_label,
            "empty_message": "No same-pollster trend data available",
            "legend_variant": "single_row",
        },
        trend_series,
        visual_paths[2],
        metadata_dir / "03_trend_visual.json",
        metadata_dir / "03_trend_visual_manifest.json",
        {
            "project_id": PROJECT_ID,
            "source_uri": polls_uri,
            "pollster": pollster,
            "trend_window_days": trend_window_days,
            "trend_start_date": trend_start_date,
            "trend_end_date": trend_end_date,
            "trend_actual_span_days": trend_actual_span_days,
            "waves": trend_waves,
        },
    )
    if trend_manifest.get("warnings"):
        raise RuntimeError(f"Trend renderer warnings: {trend_manifest['warnings']}")

    outer = [
        _render_outer(project, title=str(slide_defs["latest_support"]["title"]), visual_path=visual_paths[0], output_path=slide_paths[0]),
        _render_outer(project, title=str(slide_defs["change_since_previous_poll"]["title"]), visual_path=visual_paths[1], output_path=slide_paths[1]),
        _render_outer(project, title=str(slide_defs["recent_trend"]["title"]), visual_path=visual_paths[2], output_path=slide_paths[2]),
    ]
    methodology_manifest = render_methodology(
        _methodology_entries(latest_meta, previous_meta, comparison_days, trend_waves),
        slide_paths[3],
        title=str(slide_defs["methodology"]["title"]),
    )

    for path in slide_paths:
        _assert_slide(path)

    contact_sheet(
        [
            ("Latest poll", slide_paths[0]),
            ("Up / down", slide_paths[1]),
            ("Six-month trend", slide_paths[2]),
            ("About polling", slide_paths[3]),
        ],
        contact_dir / "four_slide_overview.jpg",
        columns=4,
    )

    caption = "\n".join(
        [
            f"Six months of {pollster} polling, using actual published poll results rather than the IPI daily model.",
            "",
            f"Latest poll: {_pretty_date(latest_date)}, n={latest_meta['sample_size']:,}.",
            f"Slide 2 compares it with the same pollster's {_pretty_date(previous_date)} wave ({comparison_days} days earlier).",
            f"Slide 3 shows all {len(trend_waves)} {pollster} polls from {_pretty_date(trend_start_date)} to {_pretty_date(trend_end_date)} inside the {trend_window_days}-day trend window; every marker is an actual published poll.",
            "",
            "These are individual survey results, so normal polling uncertainty applies. Full sampling and weighting methodology should be checked in the original pollster release.",
            "",
            f"Source: {source_label}",
            source_url,
        ]
    ).strip() + "\n"
    (period_root / "caption.txt").write_text(caption, encoding="utf-8")

    manifest = {
        "project_id": PROJECT_ID,
        "data_mode": "raw_polls_only",
        "review_state": "pending_human_review",
        "publication_enabled": False,
        "factory_reference_commit": "386b933",
        "factory_reference_workflow_run": 33894430571,
        "source_uri": polls_uri,
        "source_id": str(source_cfg.get("source_id") or "irish_polling_indicator"),
        "source_label": source_label,
        "source_url": source_url,
        "pollster": pollster,
        "latest_poll": latest_meta,
        "previous_poll": previous_meta,
        "comparison_days": comparison_days,
        "comparison_selection": comparison_selection,
        "trend_window_days": trend_window_days,
        "trend_start_date": trend_start_date,
        "trend_end_date": trend_end_date,
        "trend_actual_span_days": trend_actual_span_days,
        "trend_wave_count": len(trend_waves),
        "trend_waves": trend_waves,
        "trend_legend_default": "single_row",
        "slides": [str(path) for path in slide_paths],
        "contact_sheet": str(contact_dir / "four_slide_overview.jpg"),
        "caption": str(period_root / "caption.txt"),
        "visual_manifests": {
            "latest_support": latest_manifest,
            "change": change_manifest,
            "trend": trend_manifest,
            "methodology": methodology_manifest,
        },
        "outer_layouts": outer,
        "qa": {
            "expected_slide_count": 4,
            "actual_slide_count": len(slide_paths),
            "dimensions": [1080, 1350],
            "approved_factory_commit": "386b933",
            "source_footer_required": True,
            "model_data_used": False,
            "same_pollster_only": True,
            "comparison_and_trend_windows_independent": True,
        },
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    package = deterministic_zip(period_root, period_root / "ipi_polling_factory_review.zip")
    manifest["package"] = package
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
