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
PARTY_LABELS = {
    "FF": "Fianna Fáil",
    "FG": "Fine Gael",
    "SF": "Sinn Féin",
    "LAB": "Labour",
    "GP": "Green Party",
    "SD": "Social Democrats",
    "SPBP": "PBP-Solidarity",
    "AU": "Aontú",
    "II": "Independent Ireland",
    "OTH": "Other",
}
PARTY_CODES = ["SF", "FF", "FG", "LAB", "SD", "GP", "SPBP", "II", "AU", "OTH"]
POLL_METADATA_COLUMNS = ("date", "date_start", "date_end", "pollster", "sample_size")


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


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("IPI indicator table is empty")
    if "date" not in df.columns or "cycle" not in df.columns:
        raise RuntimeError("IPI indicator table must contain date and cycle")
    result = df.copy()
    result["_date"] = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="coerce")
    if result["_date"].isna().any():
        raise RuntimeError("IPI indicator table contains invalid dates")
    return result


def _prepare_polls(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("IPI raw polls table is empty")
    missing = [column for column in POLL_METADATA_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"IPI raw polls table is missing metadata columns: {missing}")
    result = df.copy()
    for source_col, parsed_col in (("date", "_date"), ("date_start", "_date_start"), ("date_end", "_date_end")):
        result[parsed_col] = pd.to_datetime(result[source_col], format="%Y-%m-%d", errors="coerce")
        if result[parsed_col].isna().any():
            raise RuntimeError(f"IPI raw polls table contains invalid {source_col} values")
    result["_sample_size"] = pd.to_numeric(result["sample_size"], errors="coerce")
    if result["_sample_size"].isna().any() or (result["_sample_size"] <= 0).any():
        raise RuntimeError("IPI raw polls table contains invalid sample_size values")
    return result


def _latest_and_previous(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    latest_date = df["_date"].max()
    latest_rows = df.loc[df["_date"].eq(latest_date)]
    if len(latest_rows) != 1:
        raise RuntimeError(f"Expected one latest IPI row on {latest_date.date()}, found {len(latest_rows)}")
    latest = latest_rows.iloc[0]
    cycle = str(latest["cycle"])
    earlier = df.loc[df["cycle"].astype(str).eq(cycle) & df["_date"].lt(latest_date)]
    if earlier.empty:
        raise RuntimeError(f"No previous IPI model date exists within cycle {cycle}")
    previous_date = earlier["_date"].max()
    previous_rows = earlier.loc[earlier["_date"].eq(previous_date)]
    if len(previous_rows) != 1:
        raise RuntimeError(f"Expected one previous IPI row on {previous_date.date()}, found {len(previous_rows)}")
    return latest, previous_rows.iloc[0]


def _latest_poll_metadata(polls: pd.DataFrame, latest_model_date: pd.Timestamp) -> dict[str, Any]:
    eligible = polls.loc[polls["_date"].le(latest_model_date)].copy()
    if eligible.empty:
        raise RuntimeError(f"No raw poll is available on or before model date {latest_model_date.date()}")
    latest_poll_date = eligible["_date"].max()
    latest_records = eligible.loc[eligible["_date"].eq(latest_poll_date)].copy()
    sort_columns = [column for column in ("source_row_number", "pollster") if column in latest_records.columns]
    if sort_columns:
        latest_records = latest_records.sort_values(sort_columns)
    row = latest_records.iloc[-1]
    quality_flags = str(row.get("quality_flags") or "").strip()
    return {
        "publication_date": str(row["date"]),
        "fieldwork_start": str(row["date_start"]),
        "fieldwork_end": str(row["date_end"]),
        "pollster": str(row["pollster"]),
        "sample_size": int(float(row["_sample_size"])),
        "same_publication_date_count": int(len(latest_records)),
        "quality_flags": quality_flags,
    }


def _numeric(row: pd.Series, key: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _latest_rows(latest: pd.Series, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in PARTY_CODES:
        value = _numeric(latest, code)
        if value is None:
            continue
        lower = _numeric(latest, f"{code}_lo")
        upper = _numeric(latest, f"{code}_hi")
        if lower is None or upper is None:
            raise RuntimeError(f"Missing uncertainty bounds for {code} on latest model date")
        rows.append(
            {
                "party_code": code,
                "label": PARTY_LABELS.get(code, code),
                "value": round(value * 100, 1),
                "low": round(lower * 100, 1),
                "high": round(upper * 100, 1),
            }
        )
    return sorted(rows, key=lambda item: item["value"], reverse=True)[:limit]


def _change_rows(latest: pd.Series, previous: pd.Series, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in PARTY_CODES:
        current = _numeric(latest, code)
        prior = _numeric(previous, code)
        if current is None or prior is None:
            continue
        rows.append(
            {
                "party_code": code,
                "label": PARTY_LABELS.get(code, code),
                "value": round((current - prior) * 100, 2),
                "current_pct": round(current * 100, 2),
                "previous_pct": round(prior * 100, 2),
            }
        )
    return sorted(rows, key=lambda item: item["value"], reverse=True)[:limit]


def _trend_series(df: pd.DataFrame, latest: pd.Series, *, days: int, party_limit: int) -> list[dict[str, Any]]:
    latest_date = latest["_date"]
    cycle = str(latest["cycle"])
    start = latest_date - pd.Timedelta(days=max(1, days))
    window = df.loc[
        df["cycle"].astype(str).eq(cycle)
        & df["_date"].between(start, latest_date, inclusive="both")
    ].sort_values("_date")
    ranked: list[tuple[str, float]] = []
    for code in PARTY_CODES:
        value = _numeric(latest, code)
        if value is not None:
            ranked.append((code, value))
    codes = [code for code, _ in sorted(ranked, key=lambda pair: pair[1], reverse=True)[:party_limit]]
    series: list[dict[str, Any]] = []
    for code in codes:
        points = []
        for _, row in window.iterrows():
            value = _numeric(row, code)
            if value is not None:
                points.append({"date": str(row["date"]), "value": round(value * 100, 2)})
        if points:
            series.append({"party_code": code, "label": PARTY_LABELS.get(code, code), "points": points})
    return series


def _visual_template(
    project: dict[str, Any],
    *,
    value_format: str = "percent",
    legend_variant: str | None = None,
) -> dict[str, Any]:
    render_cfg = project.get("render") or {}
    palette = render_cfg.get("palette") or {}
    params = {
        "width": 1032,
        "height": 1210,
        "max_items": int(render_cfg.get("max_items", 8)),
        "sort": "descending",
        "value_format": value_format,
        "min_visual_rows": 4,
    }
    if legend_variant:
        params["legend_variant"] = legend_variant
    return {
        "template_id": "horizontal_bar_draft_v1",
        "params": params,
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


def _methodology_entries(poll: dict[str, Any]) -> list[tuple[str, str]]:
    publication_note = (
        f"{poll['pollster']}, published {poll['publication_date']}; fieldwork {poll['fieldwork_start']} to "
        f"{poll['fieldwork_end']}; sample size {poll['sample_size']:,}."
    )
    if poll["same_publication_date_count"] > 1:
        publication_note = (
            f"The feed contains {poll['same_publication_date_count']} polls published {poll['publication_date']}. "
            f"One is {poll['pollster']}; fieldwork {poll['fieldwork_start']} to {poll['fieldwork_end']}; "
            f"sample size {poll['sample_size']:,}."
        )
    return [
        (
            "What this is",
            "The Irish Polling Indicator combines available Irish Dáil opinion polls into daily aggregated party-support estimates with 95% credible intervals.",
        ),
        ("Latest poll in the feed", publication_note),
        (
            "How the model helps",
            "It accounts for sampling error and persistent differences between polling companies, often called house effects, while allowing support to change over time.",
        ),
        (
            "What the feed does not tell us",
            "Our raw IPI feed does not include respondent geography or the full sampling and weighting design. Those details should be checked in the original pollster release.",
        ),
        (
            "How to read these slides",
            "These are modelled estimates of current support, not the result of one poll and not a prediction of the next election result.",
        ),
    ]


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    source_cfg = project.get("source") or {}
    render_cfg = project.get("render") or {}
    indicator_uri = str(source_cfg["indicator_csv"])
    polls_uri = str(source_cfg["polls_csv"])
    source_label = str(source_cfg.get("source_label") or "Irish Polling Indicator (IPI)")
    source_url = str(source_cfg.get("source_url") or "")
    source_note = f"Source: {source_label}"

    df = _prepare(_read_csv(indicator_uri))
    polls = _prepare_polls(_read_csv(polls_uri))
    latest, previous = _latest_and_previous(df)
    latest_poll = _latest_poll_metadata(polls, latest["_date"])
    limit = int(render_cfg.get("max_items", 8))
    trend_days = int(render_cfg.get("trend_days", 90))
    trend_party_limit = int(render_cfg.get("trend_party_limit", 5))
    default_legend_variant = str(render_cfg.get("trend_legend_variant") or "two_row")
    latest_rows = _latest_rows(latest, limit)
    change_rows = _change_rows(latest, previous, limit)
    trend_series = _trend_series(df, latest, days=trend_days, party_limit=trend_party_limit)
    if not latest_rows or not change_rows or not trend_series:
        raise RuntimeError("Polling factory project requires latest, change and trend data")

    latest_date = str(latest["date"])
    previous_date = str(previous["date"])
    period_key = latest_date
    period_root = output_root / f"period={period_key}"
    if period_root.exists():
        shutil.rmtree(period_root)
    slides_dir = period_root / "slides"
    assets_dir = period_root / "assets"
    variants_dir = period_root / "variants"
    metadata_dir = period_root / "metadata"
    contact_dir = period_root / "contact_sheets"
    for directory in (slides_dir, assets_dir, variants_dir, metadata_dir, contact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    slide_defs = {str(item["id"]): item for item in (project.get("slides") or {}).get("definitions", [])}
    slide_paths = [
        slides_dir / "01_latest_support.png",
        slides_dir / "02_change_since_previous_model.png",
        slides_dir / "03_recent_trend.png",
        slides_dir / "04_methodology.png",
    ]
    visual_paths = [
        assets_dir / "01_latest_support_visual.png",
        assets_dir / "02_change_visual.png",
        assets_dir / "03_trend_visual.png",
    ]
    trend_single_row_visual = variants_dir / "03_trend_single_row_visual.png"
    trend_single_row_slide = variants_dir / "03_recent_trend_single_row.png"

    latest_manifest = horizontal_bar.render(
        _visual_template(project, value_format="percent"),
        {
            "visual_id": "ipi-latest-support-factory",
            "bindings": {"label": "label", "value": "value"},
            "source_note": f"{source_note} · modelled support · {latest_date}",
            "empty_message": "No current model estimates available",
        },
        latest_rows,
        visual_paths[0],
        metadata_dir / "01_latest_support_visual.json",
        metadata_dir / "01_latest_support_visual_manifest.json",
        {
            "project_id": PROJECT_ID,
            "source_uri": indicator_uri,
            "latest_model_date": latest_date,
            "uncertainty_ranges": [{"party_code": row["party_code"], "low": row["low"], "high": row["high"]} for row in latest_rows],
        },
    )
    if latest_manifest.get("warnings"):
        raise RuntimeError(f"Approved horizontal-bar renderer warnings: {latest_manifest['warnings']}")

    change_manifest = render_diverging(
        _visual_template(project),
        {
            "visual_id": "ipi-change-factory",
            "source_note": f"{source_note} · {previous_date} → {latest_date} · model-date change",
            "empty_message": "No model-date change available",
        },
        change_rows,
        visual_paths[1],
        metadata_dir / "02_change_visual.json",
        metadata_dir / "02_change_visual_manifest.json",
        {
            "project_id": PROJECT_ID,
            "source_uri": indicator_uri,
            "previous_model_date": previous_date,
            "latest_model_date": latest_date,
        },
    )
    if change_manifest.get("warnings"):
        raise RuntimeError(f"Diverging renderer warnings: {change_manifest['warnings']}")

    trend_manifest = render_trend(
        _visual_template(project, legend_variant=default_legend_variant),
        {
            "visual_id": "ipi-trend-factory",
            "source_note": f"{source_note} · {trend_days}-day modelled trend · same election cycle",
            "empty_message": "No trend data available",
            "legend_variant": default_legend_variant,
        },
        trend_series,
        visual_paths[2],
        metadata_dir / "03_trend_visual.json",
        metadata_dir / "03_trend_visual_manifest.json",
        {
            "project_id": PROJECT_ID,
            "source_uri": indicator_uri,
            "latest_model_date": latest_date,
            "trend_days": trend_days,
            "cycle": str(latest["cycle"]),
        },
    )
    if trend_manifest.get("warnings"):
        raise RuntimeError(f"Trend renderer warnings: {trend_manifest['warnings']}")

    trend_single_row_manifest = render_trend(
        _visual_template(project, legend_variant="single_row"),
        {
            "visual_id": "ipi-trend-single-row-factory",
            "source_note": f"{source_note} · {trend_days}-day modelled trend · same election cycle",
            "empty_message": "No trend data available",
            "legend_variant": "single_row",
        },
        trend_series,
        trend_single_row_visual,
        metadata_dir / "03_trend_single_row_visual.json",
        metadata_dir / "03_trend_single_row_visual_manifest.json",
        {
            "project_id": PROJECT_ID,
            "source_uri": indicator_uri,
            "latest_model_date": latest_date,
            "trend_days": trend_days,
            "cycle": str(latest["cycle"]),
            "variant": "single_row",
        },
    )
    if trend_single_row_manifest.get("warnings"):
        raise RuntimeError(f"Single-row trend renderer warnings: {trend_single_row_manifest['warnings']}")

    outer = [
        _render_outer(project, title=str(slide_defs["latest_support"]["title"]), visual_path=visual_paths[0], output_path=slide_paths[0]),
        _render_outer(project, title=str(slide_defs["change_since_previous_model"]["title"]), visual_path=visual_paths[1], output_path=slide_paths[1]),
        _render_outer(project, title=str(slide_defs["recent_trend"]["title"]), visual_path=visual_paths[2], output_path=slide_paths[2]),
    ]
    single_row_outer = _render_outer(
        project,
        title=str(slide_defs["recent_trend"]["title"]),
        visual_path=trend_single_row_visual,
        output_path=trend_single_row_slide,
    )
    methodology_manifest = render_methodology(
        _methodology_entries(latest_poll),
        slide_paths[3],
        title=str(slide_defs["methodology"]["title"]),
    )

    for path in [*slide_paths, trend_single_row_slide]:
        _assert_slide(path)

    contact_sheet(
        [
            ("Latest estimates", slide_paths[0]),
            ("Up / down", slide_paths[1]),
            ("Trend", slide_paths[2]),
            ("How it works", slide_paths[3]),
        ],
        contact_dir / "four_slide_overview.jpg",
        columns=4,
    )
    contact_sheet(
        [("A · two-row legend", slide_paths[2]), ("B · single-row legend", trend_single_row_slide)],
        contact_dir / "trend_legend_variants.jpg",
        columns=2,
    )

    caption = "\n".join(
        [
            "Latest modelled Irish party-support estimates from the Irish Polling Indicator.",
            "",
            f"Latest model date: {latest_date}.",
            f"Slide 2 compares the previous model date ({previous_date}) with the latest model date; it is not a comparison of two individual opinion polls.",
            f"Slide 3 shows the most recent {trend_days} days within the same election cycle.",
            f"Slide 4 explains the source and includes the latest raw poll metadata available in the IPI feed: {latest_poll['pollster']}, n={latest_poll['sample_size']:,}, published {latest_poll['publication_date']}.",
            "",
            "The Irish Polling Indicator combines available polls and accounts for sampling error and pollster house effects. Its estimates are not a single opinion poll or an election forecast.",
            "",
            f"Source: {source_label}",
            source_url,
        ]
    ).strip() + "\n"
    (period_root / "caption.txt").write_text(caption, encoding="utf-8")

    manifest = {
        "project_id": PROJECT_ID,
        "review_state": "pending_human_review",
        "publication_enabled": False,
        "factory_reference_commit": "386b933",
        "factory_reference_workflow_run": 33894430571,
        "source_uri": indicator_uri,
        "polls_source_uri": polls_uri,
        "source_id": str(source_cfg.get("source_id") or "irish_polling_indicator"),
        "source_label": source_label,
        "source_url": source_url,
        "cycle": str(latest["cycle"]),
        "latest_model_date": latest_date,
        "previous_model_date": previous_date,
        "latest_poll_metadata": latest_poll,
        "trend_days": trend_days,
        "trend_legend_default": default_legend_variant,
        "slides": [str(path) for path in slide_paths],
        "variants": {
            "trend_single_row": str(trend_single_row_slide),
            "trend_legend_contact_sheet": str(contact_dir / "trend_legend_variants.jpg"),
        },
        "contact_sheet": str(contact_dir / "four_slide_overview.jpg"),
        "caption": str(period_root / "caption.txt"),
        "visual_manifests": {
            "latest_support": latest_manifest,
            "change": change_manifest,
            "trend": trend_manifest,
            "trend_single_row": trend_single_row_manifest,
            "methodology": methodology_manifest,
        },
        "outer_layouts": [*outer, single_row_outer],
        "qa": {
            "expected_slide_count": 4,
            "actual_slide_count": len(slide_paths),
            "dimensions": [1080, 1350],
            "approved_factory_commit": "386b933",
            "source_footer_required": True,
            "trend_variant_count": 2,
        },
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    package = deterministic_zip(period_root, period_root / "ipi_polling_factory_review.zip")
    manifest["package"] = package
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
