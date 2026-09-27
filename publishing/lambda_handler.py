from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from urllib import request

import boto3

SECRET_NAME = os.environ.get("INSTAGRAM_SECRET_NAME", "eirepolitic/instagram/publishing")
GRAPH_VERSION = os.environ.get("META_GRAPH_VERSION", "v25.0")
TABLE_NAME = os.environ.get("INSTAGRAM_PUBLICATION_TABLE", "eirepolitic-publications")

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


def _mark_published(table, publication_id: str, version: int, media_id: str, expected_state: str) -> None:
    table.update_item(
        Key={"pk": f"PUB#{publication_id}", "sk": "CONTROL"},
        UpdateExpression="SET #state = :published, published_media_id = :media_id",
        ConditionExpression="publication_version = :version AND #state = :expected_state",
        ExpressionAttributeNames={"#state": "state"},
        ExpressionAttributeValues={
            ":published": "published",
            ":expected_state": expected_state,
            ":version": version,
            ":media_id": media_id,
        },
    )


def _execute_publication(*, publication_id: str, expected_version: int, trigger: str) -> dict[str, object]:
    from publishing.aws_runtime import AwsInstagramPublishingRuntime
    from publishing.control import PublicationControlService
    from publishing.publisher import PublishingNeedsAttention, PublishingOutcomeUncertain

    if trigger not in {"immediate", "scheduled"}:
        return {"statusCode": 400, "body": {"error": "invalid_trigger"}}

    dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
    runtime = AwsInstagramPublishingRuntime(
        dynamodb_resource=dynamodb,
        s3_client=boto3.client("s3", region_name="us-east-2"),
        secrets_client=boto3.client("secretsmanager", region_name="us-east-2"),
        table_name=TABLE_NAME,
    )
    control = PublicationControlService(runtime.ledger)
    record = control.get(publication_id)
    if record.request.publication_version != expected_version:
        return {"statusCode": 409, "body": {"error": "publication_version_mismatch"}}
    if record.state == "published":
        return {
            "statusCode": 200,
            "body": {
                "publication_id": publication_id,
                "state": "published",
                "already_published": True,
                "published_media_id": getattr(record, "published_media_id", None),
            },
        }

    expected_state = "scheduled" if trigger == "scheduled" else "approved"
    if record.state != expected_state:
        return {
            "statusCode": 409,
            "body": {"error": "publication_not_executable", "state": record.state, "expected_state": expected_state},
        }

    last_attention = None
    for _ in range(6):
        try:
            attempt = runtime.execute(
                publication_id=publication_id,
                expected_version=expected_version,
                trigger=trigger,
                worker_id=f"instagram-{trigger}-worker",
                attempt_id=f"{publication_id}-v{expected_version}",
                now_utc=_utc_now(),
            )
        except PublishingNeedsAttention as exc:
            last_attention = str(exc)
            time.sleep(8)
            continue
        except PublishingOutcomeUncertain as exc:
            raise RuntimeError(f"Instagram publication outcome uncertain: {exc}") from exc

        if attempt.state == "published" and attempt.published_media_id:
            _mark_published(
                dynamodb.Table(TABLE_NAME),
                publication_id,
                expected_version,
                attempt.published_media_id,
                expected_state,
            )
            return {
                "statusCode": 200,
                "body": {
                    "publication_id": publication_id,
                    "state": "published",
                    "published_media_id": attempt.published_media_id,
                },
            }

    raise RuntimeError(f"Meta container did not become publishable in bounded polling window: {last_attention}")


def lambda_handler(event, context):
    event = event or {}
    action = event.get("action")

    if action == "healthcheck" or (not action and "publication_id" not in event):
        token, instagram_id = _credentials()
        account = _meta_get(f"/{instagram_id}?fields=id,username", token)
        return {
            "statusCode": 200,
            "body": {
                "meta_connected": True,
                "instagram_id_matches": str(account.get("id")) == str(instagram_id),
                "username_present": bool(account.get("username")),
                "publishing_enabled": True,
            },
        }

    if action == "execute_publication":
        return _execute_publication(
            publication_id=str(event["publication_id"]),
            expected_version=int(event["expected_version"]),
            trigger="immediate",
        )

    if "publication_id" in event and "expected_version" in event and action is None:
        return _execute_publication(
            publication_id=str(event["publication_id"]),
            expected_version=int(event["expected_version"]),
            trigger="scheduled",
        )

    return {"statusCode": 403, "body": {"error": "unsupported_publication_action"}}
