from __future__ import annotations

from typing import Any, Iterable


SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "irish_polling_indicator": {
        "display_name": "Irish Polling Indicator (IPI)",
        "required": True,
        "footer_text": "Source: Irish Polling Indicator (IPI)",
        "reference_url": "https://github.com/Irish-Polling-Indicator/ipi-data",
    },
}


def normalize_source_ids(values: Iterable[Any] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        source_id = str(value or "").strip()
        if source_id and source_id not in result:
            result.append(source_id)
    return result


def resolve_attributions(source_ids: Iterable[Any] | None) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for source_id in normalize_source_ids(source_ids):
        metadata = SOURCE_REGISTRY.get(source_id)
        if metadata is None:
            raise RuntimeError(
                f"Unknown post source_id {source_id!r}. Add it to instagram.renderer.attribution.SOURCE_REGISTRY "
                "before rendering so mandatory attribution cannot be skipped."
            )
        resolved.append({"source_id": source_id, **metadata})
    return resolved


def required_footer_text(attributions: Iterable[dict[str, Any]]) -> str:
    texts = [str(item.get("footer_text") or "").strip() for item in attributions if item.get("required")]
    return " | ".join(text for text in texts if text)
