from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from publishing.dynamodb_ledger import DynamoDBPublicationLedger
from publishing.dynamodb_runtime import DynamoDBAssetPackageStore, DynamoDBExecutionStore
from publishing.execution import acquire_execution_lease
from publishing.fingerprints import publication_request_fingerprint
from publishing.meta_http_client import MetaCredentials, MetaInstagramHttpClient
from publishing.publisher import InstagramPublisher
from publishing.validation import PublicationValidationError, validate_approval, validate_request_against_package


class RuntimeExecutionError(RuntimeError):
    pass


class S3PresignedAssetUrlProvider:
    def __init__(self, s3_client: Any, *, expires_seconds: int = 900) -> None:
        self.s3 = s3_client
        self.expires_seconds = expires_seconds

    def url_for(self, *, bucket: str, key: str) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=self.expires_seconds,
        )


class AwsInstagramPublishingRuntime:
    """Production dependency composition for deterministic Instagram execution.

    This class is intentionally not called by lambda_handler while Gate 4 is
    closed. It contains no approval bypass: execution re-validates the stored
    request, immutable asset package and approval fingerprint before Meta calls.
    """

    def __init__(
        self,
        *,
        dynamodb_resource: Any,
        s3_client: Any,
        secrets_client: Any,
        table_name: str = "eirepolitic-publications",
        secret_id: str = "eirepolitic/instagram/publishing",
        graph_version: str = "v25.0",
    ) -> None:
        table = dynamodb_resource.Table(table_name)
        self.ledger = DynamoDBPublicationLedger(table)
        self.assets = DynamoDBAssetPackageStore(table)
        self.executions = DynamoDBExecutionStore(table)
        self.asset_urls = S3PresignedAssetUrlProvider(s3_client)
        self.secrets = secrets_client
        self.secret_id = secret_id
        self.graph_version = graph_version

    def execute(
        self,
        *,
        publication_id: str,
        expected_version: int,
        trigger: Literal["scheduled", "immediate"],
        worker_id: str,
        attempt_id: str,
        now_utc: str | None = None,
    ):
        record = self.ledger.get_publication(publication_id)
        request = record.request
        if request.publication_version != expected_version:
            raise RuntimeExecutionError(
                f"publication version changed: expected {expected_version}, current {request.publication_version}"
            )
        required_state = "scheduled" if trigger == "scheduled" else "approved"
        if record.state != required_state:
            raise RuntimeExecutionError(f"publication state {record.state!r} is not valid for {trigger} execution")
        if record.approval is None:
            raise RuntimeExecutionError("publication has no approval")

        package = self.assets.get(request.asset_package_id)
        try:
            validate_request_against_package(request, package)
            fingerprint = publication_request_fingerprint(request, [asset.sha256 for asset in package.media])
            validate_approval(record.approval, request, fingerprint)
        except PublicationValidationError as exc:
            raise RuntimeExecutionError(str(exc)) from exc

        session = self.executions.open(
            publication_id=publication_id,
            publication_version=expected_version,
            attempt_id=attempt_id,
        )
        timestamp = now_utc or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        leased = acquire_execution_lease(session.attempt, owner=worker_id, now_utc=timestamp)
        session.persist(leased)

        credentials = self._load_meta_credentials()
        provider = MetaInstagramHttpClient(credentials, graph_version=self.graph_version)
        publisher = InstagramPublisher(provider, self.asset_urls, session.persist)
        return publisher.publish(request, package, leased)

    def _load_meta_credentials(self) -> MetaCredentials:
        response = self.secrets.get_secret_value(SecretId=self.secret_id)
        try:
            secret = json.loads(response["SecretString"])
            access_token = secret["page_access_token"]
            instagram_id = str(secret["instagram_account_id"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeExecutionError("Instagram publishing secret is missing required fields") from exc
        if not access_token or not instagram_id:
            raise RuntimeExecutionError("Instagram publishing secret contains empty required fields")
        return MetaCredentials(ig_user_id=instagram_id, access_token=access_token)
