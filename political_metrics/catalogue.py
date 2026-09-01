from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_CATALOGUE_DIR = Path(__file__).resolve().parents[1] / "configs" / "political_metrics" / "catalogue"


def load_catalogue_metric_ids(catalogue_dir: str | Path = DEFAULT_CATALOGUE_DIR) -> set[str]:
    """Return all unique metric IDs declared in the political metric catalogue."""
    root = Path(catalogue_dir)
    metric_ids: set[str] = set()
    duplicates: set[str] = set()
    for path in sorted(root.glob("*.yml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for metric in payload.get("metrics") or []:
            metric_id = str(metric.get("metric_id") or "").strip()
            if not metric_id:
                raise ValueError(f"catalogue metric without metric_id: {path}")
            if metric_id in metric_ids:
                duplicates.add(metric_id)
            metric_ids.add(metric_id)
    if duplicates:
        raise ValueError(f"duplicate political metric IDs across catalogue files: {sorted(duplicates)}")
    return metric_ids
