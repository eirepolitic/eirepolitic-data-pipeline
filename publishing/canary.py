from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image, ImageDraw

from publishing.control import PublicationControlService
from publishing.dynamodb_ledger import DynamoDBPublicationLedger
from publishing.dynamodb_runtime import DynamoDBAssetPackageStore
from publishing.models import InstagramOptions, PublicationRequest
from publishing.s3_assets import S3ApprovedAssetStore, SourceAsset


CANARY_PUBLICATION_ID = "instagram-pipeline-canary-20260926"
CANARY_ASSET_PACKAGE_ID = "instagram-pipeline-canary-assets-20260926"
CANARY_PROJECT_ID = "instagram-pipeline-canary"
CANARY_PERIOD = "2026-09"
CANARY_CAPTION = (
    "Eirepolitic Instagram publishing pipeline test. Temporary verification post; "
    "this will be deleted after the test. #EirepoliticTest #PipelineTest"
)
CANARY_HASHTAGS = ("#EirepoliticTest", "#PipelineTest")
CANARY_ALT_TEXT = (
    "Abstract geometric test card used only to verify the Eirepolitic Instagram publishing pipeline."
)
CANARY_ACCOUNT_REF = "eirepolitic-instagram"


@dataclass(frozen=True)
class CanaryPreparationResult:
    publication_id: str
    publication_version: int
    state: str
    asset_package_id: str
    bucket: str
    key: str
    sha256: str
    caption: str
    hashtags: tuple[str, ...]
    alt_text: str


def render_canary_source(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1080, 1350
    image = Image.new("RGB", (width, height), (245, 244, 240))
    draw = ImageDraw.Draw(image)

    draw.rectangle((110, 120, 970, 1230), fill=(18, 32, 47))
    draw.rectangle((110, 120, 970, 230), fill=(44, 132, 103))
    draw.rectangle((170, 300, 910, 1050), outline=(245, 244, 240), width=8)
    draw.rectangle((250, 400, 830, 500), fill=(245, 244, 240))
    draw.rectangle((250, 550, 620, 650), fill=(44, 132, 103))
    draw.rectangle((660, 550, 830, 650), fill=(245, 244, 240))
    draw.rectangle((250, 700, 420, 800), fill=(245, 244, 240))
    draw.rectangle((460, 700, 830, 800), fill=(44, 132, 103))
    for index, x in enumerate((250, 390, 530, 670)):
        fill = (44, 132, 103) if index % 2 == 0 else (245, 244, 240)
        draw.rectangle((x, 890, x + 100, 990), fill=fill)

    image.save(path, "PNG")
    return path


def prepare_canary(
    *,
    dynamodb_resource: Any,
    s3_client: Any,
    bucket: str,
    table_name: str = "eirepolitic-publications",
) -> CanaryPreparationResult:
    table = dynamodb_resource.Table(table_name)
    asset_store = DynamoDBAssetPackageStore(table)
    ledger = DynamoDBPublicationLedger(table)
    control = PublicationControlService(ledger)

    with TemporaryDirectory(prefix="instagram-canary-") as directory:
        source = render_canary_source(Path(directory) / "canary.png")
        package = S3ApprovedAssetStore(s3_client, bucket).finalize_package(
            project_id=CANARY_PROJECT_ID,
            period=CANARY_PERIOD,
            asset_package_id=CANARY_ASSET_PACKAGE_ID,
            sources=[SourceAsset(source, alt_text=CANARY_ALT_TEXT)],
        )

    asset_store.put(package)
    request = PublicationRequest(
        publication_id=CANARY_PUBLICATION_ID,
        publication_version=1,
        platform="instagram",
        account_ref=CANARY_ACCOUNT_REF,
        project_id=CANARY_PROJECT_ID,
        period=CANARY_PERIOD,
        asset_package_id=CANARY_ASSET_PACKAGE_ID,
        caption=CANARY_CAPTION,
        hashtags=CANARY_HASHTAGS,
        caption_mentions=(),
        instagram=InstagramOptions(
            post_type="image",
            media_tags=(),
            collaborators=(),
            location_id=None,
            first_comment=None,
        ),
    )

    try:
        record = control.create_draft(request, package)
    except Exception:
        record = control.get(CANARY_PUBLICATION_ID)
        if record.request != request:
            raise

    asset = package.media[0]
    return CanaryPreparationResult(
        publication_id=record.request.publication_id,
        publication_version=record.request.publication_version,
        state=record.state,
        asset_package_id=package.asset_package_id,
        bucket=asset.bucket,
        key=asset.key,
        sha256=asset.sha256,
        caption=record.request.caption,
        hashtags=record.request.hashtags,
        alt_text=asset.alt_text,
    )
