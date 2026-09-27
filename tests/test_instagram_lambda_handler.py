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
    fake_boto3 = types.SimpleNamespace(client=lambda service: fake_client)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    sys.modules.pop("publishing.lambda_handler", None)
    return importlib.import_module("publishing.lambda_handler")


def test_healthcheck_is_read_only(monkeypatch):
    module = _load(monkeypatch)
    monkeypatch.setattr(module, "_meta_get", lambda path, token: {"id": "123", "username": "eirepolitic"})

    result = module.lambda_handler({"action": "healthcheck"}, None)

    assert result["statusCode"] == 200
    assert result["body"] == {
        "meta_connected": True,
        "instagram_id_matches": True,
        "username_present": True,
        "publishing_enabled": False,
    }


def test_publish_actions_are_hard_blocked(monkeypatch):
    module = _load(monkeypatch)

    result = module.lambda_handler({"action": "publish"}, None)

    assert result["statusCode"] == 403
    assert result["body"]["error"] == "publishing_not_enabled"
    assert result["body"]["publishing_enabled"] is False
