from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Any

from openai import OpenAI

from process.oireachtas_speech_issue_classifier import (
    DEFAULT_MODEL,
    build_classifier_prompt,
    canonicalize_label,
    classification_schema,
)

_thread_local = threading.local()


def _client() -> OpenAI:
    value = getattr(_thread_local, "client", None)
    if value is None:
        value = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        _thread_local.client = value
    return value


def classify_row(
    row: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "low",
    verbosity: str = "low",
    max_retries: int = 8,
    max_output_tokens: int = 512,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = _client().responses.create(
                model=model,
                input=build_classifier_prompt(str(row["speech_text"])),
                reasoning={"effort": reasoning_effort},
                text={
                    "verbosity": verbosity,
                    "format": {
                        "type": "json_schema",
                        "name": "speech_issue_classification",
                        "strict": True,
                        "schema": classification_schema(),
                    },
                },
                max_output_tokens=max_output_tokens,
                store=False,
            )
            raw = str(response.output_text or "").strip()
            if not raw:
                raise ValueError("model returned empty output_text")
            payload = json.loads(raw)
            label = canonicalize_label(payload.get("issue_label"))
            if not label:
                raise ValueError(f"invalid issue label: {payload!r}")
            return {
                "status": "success",
                "speech_id": row["speech_id"],
                "issue_label": label,
                "attempts": attempt,
                "latency_seconds": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            if attempt < max_retries:
                delay = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0.0, 1.0)
                time.sleep(delay)

    return {
        "status": "failed",
        "speech_id": row["speech_id"],
        "issue_label": "",
        "attempts": max_retries,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "error": last_error,
    }
