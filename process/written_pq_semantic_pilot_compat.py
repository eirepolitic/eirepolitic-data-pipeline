#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import process.written_pq_semantic_pilot as pilot


def _strip_unsupported_json_schema_keywords(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_unsupported_json_schema_keywords(item)
            for key, item in value.items()
            if key != "uniqueItems"
        }
    if isinstance(value, list):
        return [_strip_unsupported_json_schema_keywords(item) for item in value]
    return value


_original_schema = pilot.schema


def compatible_schema(config):
    return _strip_unsupported_json_schema_keywords(_original_schema(config))


pilot.schema = compatible_schema

if __name__ == "__main__":
    raise SystemExit(pilot.main())
