from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class CaptionTemplateError(ValueError):
    pass


@dataclass(frozen=True)
class CaptionTemplate:
    template_id: str
    version: int
    required_components: tuple[str, ...]
    default_hashtags: tuple[str, ...]
    attribution: str | None
    disclaimer: str | None
    raw: dict[str, Any]


def load_caption_template(path: str | Path) -> CaptionTemplate:
    template_path = Path(path)
    if not template_path.is_file():
        raise CaptionTemplateError(f"caption template not found: {template_path}")

    data = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CaptionTemplateError("caption template must be a YAML mapping")

    template_id = str(data.get("template_id", "")).strip()
    version = data.get("version")
    if not template_id:
        raise CaptionTemplateError("template_id is required")
    if not isinstance(version, int) or version < 1:
        raise CaptionTemplateError("version must be a positive integer")

    components = data.get("components") or {}
    if not isinstance(components, dict):
        raise CaptionTemplateError("components must be a mapping")
    required = tuple(sorted(name for name, cfg in components.items() if isinstance(cfg, dict) and cfg.get("required") is True))

    hashtags = data.get("hashtags") or {}
    if not isinstance(hashtags, dict):
        raise CaptionTemplateError("hashtags must be a mapping")
    default_hashtags = tuple(str(item) for item in (hashtags.get("default") or []))
    if any(not tag.startswith("#") for tag in default_hashtags):
        raise CaptionTemplateError("all default hashtags must begin with #")

    attribution = data.get("attribution")
    disclaimer = data.get("disclaimer")
    if attribution is not None and not isinstance(attribution, str):
        raise CaptionTemplateError("attribution must be a string or null")
    if disclaimer is not None and not isinstance(disclaimer, str):
        raise CaptionTemplateError("disclaimer must be a string or null")

    return CaptionTemplate(
        template_id=template_id,
        version=version,
        required_components=required,
        default_hashtags=default_hashtags,
        attribution=attribution,
        disclaimer=disclaimer,
        raw=data,
    )


def validate_final_caption(caption: str, template: CaptionTemplate) -> None:
    if not caption.strip():
        raise CaptionTemplateError("final caption must not be empty")
    required_text = {
        "attribution": template.attribution,
        "disclaimer": template.disclaimer,
    }
    for component in template.required_components:
        text = required_text.get(component)
        if text and text not in caption:
            raise CaptionTemplateError(f"final caption is missing required {component}")
