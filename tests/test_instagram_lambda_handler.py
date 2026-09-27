from __future__ import annotations

import importlib
import json
import sys
import types


def _load(monkeypatch):
    fake_client = types.SimpleNamespace(
        get_secret_value=lambda **_: {
            "SecretString": json.dumps({"page_access_token": "token", "instagram_account_id": "123"})
        }
    )
    fake_boto3 = types.SimpleNamespace(client=lambda service, **kwargs: fake_client)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    sys.modules.pop("publishing.lambda_handler", None)
    return importlib.import_module("publishing.lambda_handler")


def test_healthcheck_reports_enabled_publishing_system(monkeypatch):
    module = _load(monkeypatch)
    monkeypatch.setattr(module, "_meta_get", lambda path, token: {"id": "123", "username": "eirepolitic"})
    result = module.lambda_handler({"action": "healthcheck"}, None)
    assert result["statusCode"] == 200
    assert result["body"]["meta_connected"] is True
    assert result["body"]["publishing_enabled"] is True


def test_immediate_publication_routes_through_generic_executor(monkeypatch):
    module = _load(monkeypatch)
    monkeypatch.setattr(
        module,
        "_execute_publication",
        lambda **kwargs: {"statusCode": 200, "body": {**kwargs, "state": "published"}},
    )
    result = module.lambda_handler(
        {"action": "execute_publication", "publication_id": "pub-1", "expected_version": 2}, None
    )
    assert result["statusCode"] == 200
    assert result["body"]["trigger"] == "immediate"
    assert result["body"]["publication_id"] == "pub-1"


def test_scheduler_payload_routes_through_same_executor(monkeypatch):
    module = _load(monkeypatch)
    monkeypatch.setattr(
        module,
        "_execute_publication",
        lambda **kwargs: {"statusCode": 200, "body": {**kwargs, "state": "published"}},
    )
    result = module.lambda_handler({"publication_id": "pub-2", "expected_version": 1}, None)
    assert result["statusCode"] == 200
    assert result["body"]["trigger"] == "scheduled"


def test_unsupported_action_is_blocked(monkeypatch):
    module = _load(monkeypatch)
    result = module.lambda_handler({"action": "publish"}, None)
    assert result == {"statusCode": 403, "body": {"error": "unsupported_publication_action"}}
