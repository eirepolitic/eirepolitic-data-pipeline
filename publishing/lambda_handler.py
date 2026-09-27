from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from urllib import request

import boto3

SECRET_NAME = os.environ.get("INSTAGRAM_SECRET_NAME", "eirepolitic/instagram/publishing")
GRAPH_VERSION = os.environ.get("META_GRAPH_VERSION", "v25.0")
GATE5_PUBLICATION_ID = "instagram-scheduled-canary-20260926"
GATE5_PUBLICATION_VERSION = 1

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _execute_gate5_scheduled_canary(event: dict[str, object]) -> dict[str, object]:
    from publishing.aws_runtime import AwsInstagramPublishingRuntime
    from publishing.publisher import PublishingNeedsAttention, PublishingOutcomeUncertain

    if (
        event.get("publication_id") != GATE5_PUBLICATION_ID
        or int(event.get("expected_version", -1)) != GATE5_PUBLICATION_VERSION
    ):
        return {"statusCode": 403, "body": {"error": "scheduled_canary_not_authorized"}}

    dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
    runtime = AwsInstagramPublishingRuntime(
        dynamodb_resource=dynamodb,
        s3_client=boto3.client("s3", region_name="us-east-2"),
        secrets_client=boto3.client("secretsmanager", region_name="us-east-2"),
    )

    last_attention = None
    for _ in range(6):
        try:
            attempt = runtime.execute(
                publication_id=GATE5_PUBLICATION_ID,
                expected_version=GATE5_PUBLICATION_VERSION,
                trigger="scheduled",
                worker_id="gate5-scheduler-worker-20260926",
                attempt_id="gate5-scheduler-attempt-20260926",
                now_utc=_utc_now(),
            )
        except PublishingNeedsAttention as exc:
            last_attention = str(exc)
            time.sleep(8)
            continue
        except PublishingOutcomeUncertain as exc:
            return {
                "statusCode": 409,
                "body": {
                    "publication_id": GATE5_PUBLICATION_ID,
                    "state": "outcome_uncertain",
                    "detail": str(exc),
                },
            }

        if attempt.state == "published" and attempt.published_media_id:
            table = dynamodb.Table("eirepolitic-publications")
            table.update_item(
                Key={"pk": f"PUB#{GATE5_PUBLICATION_ID}", "sk": "CONTROL"},
                UpdateExpression="SET #state = :published, published_media_id = :media_id",
                ConditionExpression="publication_version = :version AND #state = :scheduled",
                ExpressionAttributeNames={"#state": "state"},
                ExpressionAttributeValues={
                    ":published": "published",
                    ":scheduled": "scheduled",
                    ":version": GATE5_PUBLICATION_VERSION,
                    ":media_id": attempt.published_media_id,
                },
            )
            return {
                "statusCode": 200,
                "body": {
                    "publication_id": GATE5_PUBLICATION_ID,
                    "state": "published",
                    "published_media_id": attempt.published_media_id,
                },
            }

    raise RuntimeError(f"Meta container did not become publishable during scheduled canary: {last_attention}")


def lambda_handler(event, context):
    """AWS entrypoint with a single Gate 5 scheduled-canary exception."""
    event = event or {}
    if event.get("publication_id") == GATE5_PUBLICATION_ID:
        return _execute_gate5_scheduled_canary(event)

    token, instagram_id = _credentials()
    action = event.get("action", "healthcheck")

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

    return {
        "statusCode": 403,
        "body": {
            "error": "publishing_not_enabled",
            "publishing_enabled": False,
        },
    }
