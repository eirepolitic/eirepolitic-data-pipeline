from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from urllib import parse, request

import boto3

SECRET_ID = os.getenv("INSTAGRAM_SECRET_ID", "eirepolitic/instagram/publishing")
GRAPH_VERSION = os.getenv("INSTAGRAM_GRAPH_VERSION", "v25.0")
GRAPH_BASE_URL = os.getenv("INSTAGRAM_GRAPH_BASE_URL", "https://graph.facebook.com")
CANARY_ACTION = "gate4_canary_20260926"
CANARY_PUBLICATION_ID = "instagram-pipeline-canary-20260926"
CANARY_PUBLICATION_VERSION = 1


def _load_secret() -> dict[str, Any]:
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=SECRET_ID)
    secret = json.loads(response["SecretString"])
    if not secret.get("page_access_token") or not secret.get("instagram_account_id"):
        raise RuntimeError("Instagram publishing secret is missing required fields")
    return secret


def _meta_get(path: str, access_token: str) -> dict[str, Any]:
    url = f"{GRAPH_BASE_URL.rstrip('/')}/{GRAPH_VERSION}/{path.lstrip('/')}"
    req = request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    with request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _execute_gate4_canary() -> dict[str, Any]:
    from publishing.aws_runtime import AwsInstagramPublishingRuntime
    from publishing.control import PublicationControlService
    from publishing.publisher import PublishingNeedsAttention, PublishingOutcomeUncertain

    dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
    runtime = AwsInstagramPublishingRuntime(
        dynamodb_resource=dynamodb,
        s3_client=boto3.client("s3", region_name="us-east-2"),
        secrets_client=boto3.client("secretsmanager", region_name="us-east-2"),
    )
    control = PublicationControlService(runtime.ledger)
    record = control.get(CANARY_PUBLICATION_ID)
    if record.request.publication_version != CANARY_PUBLICATION_VERSION:
        return {"statusCode": 409, "body": {"error": "canary_version_mismatch"}}

    if record.state == "draft":
        package = runtime.assets.get(record.request.asset_package_id)
        record = control.approve(
            CANARY_PUBLICATION_ID,
            package,
            approval_id="gate4-canary-approval-20260926",
            approved_by="user-explicit-gate4-chat",
            approved_at_utc=_utc_now(),
        )
    if record.state == "published":
        return {
            "statusCode": 200,
            "body": {
                "publication_id": CANARY_PUBLICATION_ID,
                "state": "published",
                "already_published": True,
                "published_media_id": getattr(record, "published_media_id", None),
            },
        }
    if record.state != "approved":
        return {"statusCode": 409, "body": {"error": "canary_not_approved", "state": record.state}}

    try:
        attempt = runtime.execute(
            publication_id=CANARY_PUBLICATION_ID,
            expected_version=CANARY_PUBLICATION_VERSION,
            trigger="immediate",
            worker_id="gate4-canary-worker-20260926",
            attempt_id="gate4-canary-attempt-20260926",
        )
    except PublishingNeedsAttention as exc:
        return {
            "statusCode": 202,
            "body": {
                "publication_id": CANARY_PUBLICATION_ID,
                "state": "waiting_for_meta",
                "retry_same_canary": True,
                "detail": str(exc),
            },
        }
    except PublishingOutcomeUncertain as exc:
        return {
            "statusCode": 409,
            "body": {
                "publication_id": CANARY_PUBLICATION_ID,
                "state": "outcome_uncertain",
                "retry_same_canary": False,
                "detail": str(exc),
            },
        }

    if attempt.state != "published" or not attempt.published_media_id:
        return {"statusCode": 409, "body": {"error": "canary_not_published", "state": attempt.state}}

    table = dynamodb.Table("eirepolitic-publications")
    table.update_item(
        Key={"pk": f"PUB#{CANARY_PUBLICATION_ID}", "sk": "CONTROL"},
        UpdateExpression="SET #state = :published, published_media_id = :media_id",
        ConditionExpression="publication_version = :version AND #state = :approved",
        ExpressionAttributeNames={"#state": "state"},
        ExpressionAttributeValues={
            ":published": "published",
            ":approved": "approved",
            ":version": CANARY_PUBLICATION_VERSION,
            ":media_id": attempt.published_media_id,
        },
    )
    return {
        "statusCode": 200,
        "body": {
            "publication_id": CANARY_PUBLICATION_ID,
            "state": "published",
            "published_media_id": attempt.published_media_id,
        },
    }


def lambda_handler(event, context):
    action = (event or {}).get("action", "healthcheck")
    if action == CANARY_ACTION:
        return _execute_gate4_canary()
    if action != "healthcheck":
        return {
            "statusCode": 403,
            "body": {
                "error": "publishing_not_enabled",
                "publishing_enabled": False,
            },
        }

    secret = _load_secret()
    instagram_id = str(secret["instagram_account_id"])
    profile = _meta_get(
        f"{parse.quote(instagram_id, safe='')}?fields=id,username",
        secret["page_access_token"],
    )
    return {
        "statusCode": 200,
        "body": {
            "meta_connected": True,
            "instagram_id_matches": str(profile.get("id")) == instagram_id,
            "username_present": bool(profile.get("username")),
            "publishing_enabled": False,
        },
    }
