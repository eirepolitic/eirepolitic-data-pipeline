# Step 12 — Asset Readiness and Media Hosting

Status: **complete**

Research date: 2026-09-04

Scope: define when generated Instagram assets are safe to enter the publication system, how immutable approved assets should be stored, and how Meta should retrieve them at execution time.

No S3 configuration, asset conversion, public access, presigned URL, Meta container, or production publishing path was created.

---

## Short conclusion

The current preview/output pipeline is useful for review, but it should **not** be used directly as the production publication asset store.

Recommended model:

```text
generated preview assets
      ↓
human/QA review
      ↓
finalize publication assets
      ↓
immutable approved AssetPackage in private S3
      ↓
publication-time retrieval URL
      ↓
Meta downloads media
```

Important findings:

1. The current renderer writes PNG files, while Meta's current Content Publishing guide says JPEG is the supported image format for the documented image publishing path. A publication-finalization step is therefore required before direct API publishing.
2. Approved assets should use immutable S3 keys and recorded SHA-256 hashes; never publish from the mutable `latest/` preview prefix.
3. S3 Versioning is useful additional protection against accidental overwrite/deletion, but the application should still use immutable asset-package paths rather than intentionally overwriting objects.
4. Meta cURLs the supplied media URL at the time of the publishing/container-creation attempt; the URL must be reachable without interactive authentication then.
5. For direct Meta, the preferred design is to keep canonical S3 objects private and generate a short-lived retrieval URL only immediately before creating Meta containers. This pattern should be canary-tested because Meta documents the public-reachability requirement but does not specifically prescribe S3 presigned URLs.
6. Temporary GitHub Actions artifact URLs and the repo's mutable S3 `latest/` preview URLs must not become production publication dependencies.

---

# 1. Existing repo asset flow

The current Instagram renderer:

```text
process/instagram_render_campaign.py
```

writes rendered images under:

```text
generated_posts/<campaign>/png/*.png
```

The review workflow then records:

- `review_status`;
- `publish_ready`;
- render warnings;
- review metadata.

The S3 preview uploader:

```text
process/instagram_upload_preview_to_s3.py
```

currently uploads review output to both:

```text
instagram/previews/<campaign>/<run_label>/...
```

and a mutable:

```text
instagram/previews/<campaign>/latest/...
```

The script itself states that `latest` is overwritten for each preview upload.

### Consequence

These paths are appropriate for review, but **not** as the canonical production publication identity.

The publication system must never say:

```text
publish whatever is currently under latest/
```

because that allows an approved schedule to silently pick up a newer render.

---

# 2. Current Meta media-retrieval requirement

Meta's current official Content Publishing documentation says it cURLs media used in publishing attempts, so the media must be hosted on a publicly accessible server at the time of the attempt.

For image/video container creation the API receives:

```text
image_url
```

or:

```text
video_url
```

and Meta retrieves the media from that URL.

Meta's current documentation also says an unpublished container expires after 24 hours.

Source:

- Meta current Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

### Architecture consequence

The stable publication asset is the **object**, not the temporary URL.

Use:

```text
canonical private S3 object
     ↓ at execution time
temporary Meta-retrievable HTTPS URL
     ↓
create container
```

Do not store the temporary URL as the permanent asset identity in `PublicationRequest`.

---

# 3. Current format mismatch: PNG vs Meta JPEG requirement

The repo currently produces:

```text
*.png
```

for Instagram campaign renders.

Meta's current general Content Publishing guide says:

> JPEG is the only image format supported.

It explicitly excludes extended JPEG variants such as MPO/JPS.

Source:

- Meta current Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

### Required future change

Do not change the existing renderer merely to satisfy publishing.

Instead introduce a future deterministic finalization step conceptually like:

```text
reviewed PNG render
       ↓
publication finalizer
       ↓
JPEG delivery asset
       ↓
quality/dimensions/hash verified
       ↓
immutable AssetPackage
```

This keeps review/render source assets distinct from platform-delivery assets.

### Why this is preferable

If another platform later accepts PNG directly, Eirepolitic can keep the high-quality/source render while deriving an Instagram-specific delivery asset.

The publication data model can record both relationships without changing the original render.

---

# 4. Approved asset package requirements

An `AssetPackage` should not become `publication_ready=true` unless all required validations pass.

Minimum recommended checks:

```text
✓ project ID matches expected project
✓ period matches expected period
✓ expected media count known
✓ actual media count matches expected count
✓ media order explicit
✓ every file exists
✓ every file is readable
✓ MIME type validated from actual output, not filename alone
✓ file format supported by selected platform adapter
✓ width/height recorded
✓ aspect ratio allowed by post type/policy
✓ file size recorded
✓ SHA-256 recorded
✓ alt text present where required by policy
✓ renderer/QA warnings resolved
✓ human visual review approved
✓ source/version metadata recorded
✓ no mutable latest URL is used as canonical identity
```

Platform-specific validation happens before package publication readiness or in a derived platform delivery package, depending on final implementation.

---

# 5. Dimensions and aspect-ratio policy

Meta's high-level current guide does not provide one universal image width/height pair that Eirepolitic should blindly hard-code for every feed format.

It does document carousel behaviour in which carousel images are cropped based on the first image's orientation, with a default 1:1 behaviour in the guide.

Source:

- Meta Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

### Recommended validation approach

The future adapter should have versioned platform policy, for example:

```yaml
instagram_policy:
  api_version: ...
  image:
    allowed_formats:
      - image/jpeg
    allowed_aspect_ratios: ...
    max_file_size: ...
```

Exact limits should come from the current endpoint/version documentation at implementation time and be covered by canary tests.

Do not assume that a visually acceptable Instagram app upload is necessarily valid for the API endpoint.

---

# 6. Carousel readiness

For an eight-slide carousel, the `AssetPackage` should explicitly contain:

```yaml
expected_media_count: 8

media:
  - ordinal: 1
    asset_id: slide_01
    ...
  - ordinal: 2
    asset_id: slide_02
    ...
  ...
  - ordinal: 8
    asset_id: slide_08
```

At final publication validation:

```text
expected count == actual count
```

and each ordered asset hash must match the approved package.

Do not infer order from:

- filesystem enumeration;
- alphabetical S3 listing;
- GitHub artifact listing;
- Meta response ordering.

This protects against incomplete/reordered carousels.

---

# 7. Recommended S3 canonical path

Conceptual future path:

```text
s3://eirepolitic-data/
  instagram/
    approved/
      <project_id>/
        <period>/
          <asset_package_id>/
            asset_package.json
            media/
              01.jpg
              02.jpg
              03.jpg
```

Example:

```text
s3://eirepolitic-data/instagram/approved/party_speech_breakdown/2026-08/asset_01J.../media/01.jpg
```

### Rules

- `asset_package_id` path is immutable;
- never overwrite media within an approved package;
- a corrected render creates a new asset package ID;
- no `latest/` alias is used for execution;
- source render and platform delivery derivative may both be referenced in package metadata.

---

# 8. S3 Versioning

AWS S3 Versioning preserves multiple object versions and allows recovery from accidental overwrite/deletion.

AWS states that when Versioning is enabled, overwriting an object creates a new version and the previous version remains recoverable.

Source:

- AWS S3 Versioning: https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html

### Recommendation

Enable/retain S3 Versioning for the bucket or publication asset area if operationally appropriate.

But Versioning is **defence in depth**, not permission to use mutable asset keys.

Preferred model remains:

```text
immutable key + SHA256 + optional S3 version ID
```

rather than:

```text
same key overwritten repeatedly, rely on S3 version history to figure out which one was approved
```

---

# 9. Canonical object identity

Recommended `AssetPackage` media identity:

```yaml
object:
  bucket: eirepolitic-data
  key: instagram/approved/.../01.jpg
  version_id: optional-if-versioning-enabled

sha256: ...
```

The stable identity is therefore:

```text
bucket + key + optional S3 version ID + expected SHA256
```

AWS documents that bucket + key + optional version ID uniquely identifies an S3 object/version.

Source:

- AWS S3 overview: https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html

---

# 10. Why hashes are required even with immutable paths

A SHA-256 hash allows the publisher to prove:

```text
the bytes being published == the bytes that were approved
```

Before container creation, the worker should compare the canonical object's current identity/checksum against the `AssetPackage` expectation.

If the bytes differ:

```text
validation failure
→ needs_attention
→ do not publish
```

This protects against:

- accidental overwrite;
- wrong-object reference;
- packaging bugs;
- corrupted uploads;
- stale manifests.

---

# 11. Presigned URLs for direct Meta

AWS S3 presigned URLs provide time-limited access to otherwise private S3 objects.

AWS documents that:

- the URL is valid for the configured lifetime;
- SDK/CLI presigned URLs can be configured for up to seven days in some credential scenarios;
- a presigned URL expires earlier if the underlying temporary credential expires/revokes first.

Source:

- AWS S3 presigned URLs: https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html

### Proposed direct-Meta pattern

```text
private immutable S3 object
       ↓
Lambda executes at publication time
       ↓
generate presigned GET URL
       ↓
Meta cURLs URL immediately to create container
       ↓
URL later expires
```

### Important qualification

Meta's documentation requires the media URL to be publicly reachable at the publishing attempt but does **not** specifically document or certify S3 presigned URLs.

A presigned URL is technically reachable without interactive login while valid, so it is a strong candidate, but the exact approach must be validated in the future non-production Meta canary.

Do not treat this research conclusion as proof until Meta successfully retrieves a canary asset.

---

# 12. Presigned URL lifetime

Do not make the URL excessively short.

The worker may need time for:

- Meta to fetch the object;
- network delay;
- a bounded retry of container creation;
- media processing/reconciliation.

A future implementation could begin with a conservative retrieval lifetime such as tens of minutes to around an hour, then validate actual behaviour in the canary.

The precise number should be selected during implementation based on:

- Lambda/role credential lifetime;
- actual Meta fetch timing;
- retry window;
- exposure minimization.

Do not hard-code a lifetime based solely on the theoretical S3 maximum.

---

# 13. Presigned URLs are credentials

Although they do not contain the Meta token, a valid S3 presigned URL grants temporary access to the object.

Therefore it should be treated as sensitive operational data.

Do not store/log full presigned URLs in:

- publication ledger;
- High Director conversation;
- CloudWatch structured logs;
- GitHub artifacts;
- error messages.

Log only safe metadata such as:

```text
bucket
key
version ID
URL generated at
URL expiry timestamp
```

if needed.

---

# 14. Alternative: permanently/publicly served assets

A second possible design is to expose approved assets through a stable public S3/CloudFront URL.

Advantages:

- simple for external schedulers such as Buffer;
- no concern about presigned URL expiry.

Disadvantages:

- media remains publicly accessible before publication;
- public delivery policy/CDN permissions must be managed;
- harder to keep unpublished assets private;
- increased chance of accidental discovery/sharing.

### Recommendation

For **direct Meta**, prefer private canonical storage + execution-time temporary URL if canary validation succeeds.

If **Buffer hybrid** is selected, Step 8 found that Buffer requires stable publicly reachable media URLs until the future scheduled time, so a public/stable delivery layer may become necessary.

This asset-hosting difference remains an important direct-vs-hybrid trade-off.

---

# 15. GitHub Actions artifacts

GitHub Actions artifacts are useful for:

- render outputs;
- QA/review;
- debugging;
- temporary CI transfer.

They are not appropriate as canonical publication hosting because:

- they are workflow-oriented temporary artifacts;
- URLs/access behaviour are not designed as stable third-party media-delivery endpoints;
- retention can change/expire;
- publication should not depend on a past workflow-run artifact still existing.

Approved production media should be promoted/finalized into object storage explicitly.

---

# 16. Review preview versus approved publication asset

Recommended distinction:

```text
instagram/previews/...
```

means:

> content intended for human review.

```text
instagram/approved/...
```

means:

> immutable content package that passed the publication asset gate.

Promotion/finalization should be a deliberate deterministic action after review.

Do not automatically treat every preview upload as publication-ready.

---

# 17. Proposed finalization step

Conceptual future command/module only — not implemented in this research:

```text
instagram_finalize_asset_package.py
```

Responsibilities could include:

1. read approved review manifest;
2. verify review status and safety gates;
3. load expected source renders;
4. create Instagram-compatible JPEG derivatives;
5. verify output dimensions/aspect ratio/policy;
6. calculate SHA-256;
7. write immutable `asset_package.json`;
8. upload to immutable approved S3 prefix;
9. verify uploaded object metadata/hash;
10. mark package publication-ready.

It should **not**:

- publish to Instagram;
- create Meta containers;
- schedule jobs;
- create publication approval.

---

# 18. AssetPackage example

```yaml
asset_package_id: asset_01J...
schema_version: 1
project_id: party_speech_breakdown
period: "2026-08"

source:
  repository_commit: ...
  render_run_id: ...

media:
  - asset_id: slide_01
    ordinal: 1
    source_render:
      format: image/png
      sha256: ...
    delivery_asset:
      bucket: eirepolitic-data
      key: instagram/approved/party_speech_breakdown/2026-08/asset_01J.../media/01.jpg
      version_id: ...
      mime_type: image/jpeg
      width: ...
      height: ...
      size_bytes: ...
      sha256: ...
    alt_text: "..."

readiness:
  expected_media_count: 8
  actual_media_count: 8
  qa_status: passed
  human_visual_review_status: approved
  publication_ready: true
```

No HTTP delivery URL is permanently stored here.

---

# 19. Execution-time asset validation

Immediately before publishing, the worker should verify:

```text
publication request references approved AssetPackage
AssetPackage publication_ready = true
project/period match publication request
media count/order match publication request
object exists
object version/identity is expected
SHA-256/checksum matches
format/MIME is supported
configured dimensions/aspect rules still pass
```

Then generate the temporary retrieval URL.

If any check fails, **do not create a Meta container**.

This ensures a broken asset package fails before Instagram is touched.

---

# 20. Media URL failure behaviour

If Meta cannot retrieve a URL during container creation:

1. do not modify the approved publication request;
2. verify the canonical S3 object still exists and matches its hash;
3. generate a fresh retrieval URL if the prior one may have expired;
4. retry only within the bounded safe execution policy;
5. record the sanitized failure in `ExecutionAttempt`;
6. never switch automatically to a different/mutable media object.

This keeps retries deterministic.

Detailed retry rules are Step 16.

---

# 21. S3 access policy principle

Canonical approved assets should remain private by default.

The future publication worker should receive only the minimum permissions required, conceptually:

```text
s3:GetObject
```

for approved publication prefixes, plus whatever metadata/checksum calls are actually required.

It should not need broad write/delete access to approved assets merely to publish them.

The final IAM/secrets policy will be part of implementation planning, not this research step.

---

# 22. Asset retention

Published asset packages should not be immediately deleted after posting.

The publication ledger should remain able to answer:

```text
Exactly which bytes were published?
```

Recommended policy direction:

- retain approved publication assets for the lifetime of the publication ledger/history;
- use normal S3 lifecycle/storage-tier policies later if storage becomes material;
- do not delete package assets solely because the GitHub workflow artifact expired.

At Eirepolitic's static-image volume, storage cost should be small.

---

# 23. Direct Meta versus Buffer asset handling

| Area | Direct Meta | Buffer hybrid |
|---|---|---|
| Canonical storage | Private S3 recommended | Private canonical storage still possible |
| Delivery URL created | At execution time | Normally when Buffer scheduled post is created |
| URL lifetime required | Short, if Meta fetches immediately | Must remain reachable until scheduled publication |
| Presigned S3 URL fit | Strong candidate; canary required | Buffer explicitly warns against expiring signed URLs for future jobs |
| Pre-publication asset exposure | Low | Higher if stable public URL required |
| Asset-hosting complexity | Low | Moderate |

This continues to favor direct Meta where keeping unpublished assets private is desired.

---

# 24. Step 12 verdict

Recommended asset model:

```text
existing renderer PNG
       ↓
human review
       ↓
deterministic publication finalizer
       ↓
Instagram-compatible JPEG derivative
       ↓
SHA-256 + metadata + ordered immutable AssetPackage
       ↓
private versioned S3 approved prefix
       ↓
publication-time temporary retrieval URL
       ↓
Meta
```

Key rules:

1. Never publish from `latest/`.
2. Never use GitHub Actions artifact URLs for production publishing.
3. Do not assume current PNG renders are directly publishable through Meta; finalize JPEG delivery assets.
4. Record explicit order/count/dimensions/MIME/file size/SHA-256.
5. Approved asset packages are immutable; corrections create new IDs.
6. Keep canonical S3 objects private by default.
7. Generate retrieval URLs only at execution time for direct Meta.
8. Treat presigned URLs as temporary credentials and never log them.
9. Canary-test Meta retrieval from the chosen temporary URL mechanism before production enablement.
10. Retain the published asset package so the ledger can prove exactly what bytes were delivered.

---

## Sources

### Meta

- Current Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672
  - media must be reachable on a public server at publishing attempt;
  - Meta cURLs `image_url` / `video_url`;
  - current general image limitation is JPEG;
  - carousel/container behaviour.

### AWS

- S3 presigned URL documentation: https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html
- S3 Versioning: https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html
- S3 object/key/version identity overview: https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html

### Repository

- `process/instagram_render_campaign.py`
- `process/instagram_upload_preview_to_s3.py`

---

## Confidence / unresolved items

**High confidence:**

- current renderer produces PNG outputs;
- current Meta general Content Publishing guide requires JPEG for image publishing;
- Meta retrieves media by URL from a publicly reachable server at the publishing attempt;
- mutable `latest/` preview paths are unsuitable as canonical approved publication assets;
- immutable S3 objects + hashes are appropriate for publication auditability;
- S3 Versioning adds recovery protection;
- direct Meta has a materially cleaner private-asset model than Buffer's current scheduled-media URL requirement.

**Must be proven during future canary implementation:**

- Meta successfully cURLs the exact S3 presigned-URL pattern generated by the future runtime;
- suitable presigned URL lifetime under the actual Lambda/IAM credential type;
- exact image dimensions/aspect/file-size policy for the selected current Graph API version;
- JPEG encoding/quality settings that preserve visual quality and text readability;
- whether any platform-specific derivative pipeline is needed beyond Instagram JPEG conversion.

**Next research step:**

Step 13 will design reusable caption templates/defaults while preserving the rule that the exact final approved caption is explicitly stored before publication.
