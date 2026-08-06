from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "configs/oireachtas/speech_model_candidates.yml"
PRICING_PATH = ROOT / "configs/oireachtas/speech_model_pricing_verified_2026-08-05.csv"
CLASSIFIER_WORKFLOW = ROOT / ".github/workflows/oireachtas_speech_issue_classifier_v2.yml"
ORCHESTRATOR_WORKFLOW = ROOT / ".github/workflows/oireachtas_refresh_validation_orchestrator.yml"


def test_candidate_snapshot_has_two_unselected_account_unverified_models() -> None:
    payload = yaml.safe_load(CANDIDATES_PATH.read_text(encoding="utf-8"))
    assert payload["selection"]["selected_model"] is None
    assert payload["selection"]["status"] == "requires_reviewed_evaluation_and_user_approval"

    candidates = payload["candidates"]
    assert [item["model_name"] for item in candidates] == [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
    ]
    for candidate in candidates:
        assert candidate["account_availability"] == "not_verified"
        assert candidate["responses_api"] is True
        assert candidate["batch_api"] is True
        assert candidate["structured_outputs"] is True


def test_verified_pricing_matches_candidate_batch_rates() -> None:
    payload = yaml.safe_load(CANDIDATES_PATH.read_text(encoding="utf-8"))
    pricing = pd.read_csv(PRICING_PATH).set_index("model_name")

    for candidate in payload["candidates"]:
        model = candidate["model_name"]
        assert pricing.loc[model, "input_price_per_million"] == candidate[
            "batch_input_price_per_million"
        ]
        assert pricing.loc[model, "output_price_per_million"] == candidate[
            "batch_output_price_per_million"
        ]
        assert candidate["batch_input_price_per_million"] == candidate[
            "standard_input_price_per_million"
        ] / 2
        assert candidate["batch_output_price_per_million"] == candidate[
            "standard_output_price_per_million"
        ] / 2


def test_workflows_do_not_hardcode_or_auto_select_candidate_model() -> None:
    classifier = CLASSIFIER_WORKFLOW.read_text(encoding="utf-8")
    orchestrator = ORCHESTRATOR_WORKFLOW.read_text(encoding="utf-8")

    assert "gpt-5.6-luna" not in classifier
    assert "gpt-5.6-terra" not in classifier
    assert "gpt-5.6-luna" not in orchestrator
    assert "gpt-5.6-terra" not in orchestrator
    assert "OIREACHTAS_SPEECH_CLASSIFIER_MODEL" in orchestrator
    assert "Model unavailable" not in classifier
    assert "verify-models" in classifier
