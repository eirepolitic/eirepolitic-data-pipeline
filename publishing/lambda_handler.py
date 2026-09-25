from __future__ import annotations

import json
import os
from urllib import request

import boto3

SECRET_NAME = os.environ.get("INSTAGRAM_SECRET_NAME", "eirepolitic/instagram/publishing")
GRAPH_VERSION = os.environ.get("META_GRAPH_VERSION", "v25.0")

secrets = boto3.client("secretsmanager")


def _credentials() -> tuple[str, str]:
    response = secrets.get_secret_value(SecretId=SECRET_NAME)
    value = json.loads(response["SecretString"])
    return value["page_access_token"], value["instagram_account_id"]


def _meta_get(path: str, token: str) -> dict[str, object]:
    req = request.Request(
        f"https://graph.facebook.com/{GRAPH_VERSION}{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    with request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def lambda_handler(event, context):
    """AWS entrypoint. Publishing stays hard-gated until the canary gate is approved."""
    token, instagram_id = _credentials()
    action = (event or {}).get("action", "healthcheck")

    if action == "healthcheck":
        account = _meta_get(f"/{instagram_id}?fields=id,username", token)
        return {
            "statusCode": 200,
            "body": {
                "meta_connected": True,
                "instagram_id_matches": str(account.get("id")) == str(instagram_id),
                "username_present": bool(account.get("username")),
                "publishing_enabled": False,
            },
        }

    # Gate 4: no /media or /media_publish calls are reachable from this handler yet.
    return {
        "statusCode": 403,
        "body": {
            "error": "publishing_not_enabled",
            "publishing_enabled": False,
        },
    }
