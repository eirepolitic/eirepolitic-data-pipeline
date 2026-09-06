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
_original_classify = pilot.classify


def compatible_schema(config):
    return _strip_unsupported_json_schema_keywords(_original_schema(config))


def normalized_classify(client, config, record, model):
    result, usage = _original_classify(client, config, record, model)
    expected = [str(q["question_id"]) for q in record["questions"]]
    first_by_id = {}
    for item in result.get("questions", []):
        qid = str(item.get("question_id", ""))
        if qid in expected and qid not in first_by_id:
            first_by_id[qid] = item
    result["questions"] = [first_by_id[qid] for qid in expected if qid in first_by_id]
    return result, usage


pilot.schema = compatible_schema
pilot.classify = normalized_classify

if __name__ == "__main__":
    raise SystemExit(pilot.main())
