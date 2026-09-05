# Instagram Publishing — Detailed Implementation Plan

Status: **planning complete — implementation not started**

Date: 2026-09-04

Target architecture approved by user:

```text
High Director / Overlord
        ↓
Publication control plane
        ↓
DynamoDB publication ledger
        ↓
EventBridge Scheduler
        ↓
Lambda deterministic publisher
        ↓
Meta Instagram API
        ↓
Instagram
```

Supporting services:

```text
S3              immutable approved publication assets
Secrets Manager Meta runtime credential
SQS             scheduler DLQ
CloudWatch       metrics/logs/alarms
SNS/User Notifications operator alerts
GitHub OIDC      short-lived AWS deployment credentials
```

Approved v1 scope:

- existing Instagram Professional account;
- static image + carousel;
- scheduled + immediate publishing;
- exact captions;
- hashtags and caption mentions;
- media tags;
- alt text;
- collaborators/location if canary validation confirms current Meta support for the account/app route;
- optional first comment if canary validation confirms behaviour;
- cancel/reschedule;
- status/history queries through High Director;
- human approval required for every live publication.

No live configuration, credentials, scheduler, infrastructure or publication should be created until the relevant implementation gate is explicitly approved.

---

# 1. Implementation principles

1. High Director controls intent and human approval only.
2. No LLM waits for a future time or acts as a background scheduler.
3. The publication worker is deterministic and sends only already-approved content.
4. Existing `publish_ready`/review fields mean content-ready, not Instagram-authorized.
5. Approved publication versions are immutable.
6. Every external side effect is idempotent/reconcilable.
7. Production Meta credentials never enter GitHub Actions or High Director context.
8. Instagram-specific execution details remain behind an adapter boundary.
9. No automatic destructive delete in v1.
10. Live publication remains disabled until the canary gate passes.

---

# 2. Account/API discovery gate

The account is already Professional, so do **not** perform an automatic conversion.

First determine and record:

```text
Professional subtype: Business or Creator
Current linked Facebook Page: yes/no
Current Meta Business ownership/access
Current account privacy/public status
Current account ID/handle
```

### Decision rule

Preferred direct route remains:

```text
Instagram API with Facebook Login
```

because the desired feature set includes richer tagging/collaboration capability.

If the existing account is already Business and linked correctly:

```text
no account-type change required
```

If it is Creator:

```text
do not switch automatically
```

First verify whether the required v1 features work for the current Creator account through the selected API route. Only propose a Business switch if a required feature genuinely depends on it.

### Gate 1 approval required before:

- changing Professional subtype;
- creating/linking a Facebook Page;
- creating/configuring a Meta App;
- connecting the live Instagram account.

---

# 3. Proposed repository structure

Research recommendation only; create during implementation as needed.

```text
instagram/
  caption_templates/
    member_profile.yml
    party_speech_breakdown.yml
  publication_policies/
    instagram.yml

publishing/
  __init__.py
  models/
    asset_package.py
    publication_request.py
    publication_approval.py
    publication_schedule.py
    execution_attempt.py
    published_media.py
  control/
    service.py
    validation.py
    fingerprints.py
    timezone.py
  assets/
    finalizer.py
    s3_store.py
  providers/
    base.py
    instagram_meta.py
  scheduling/
    eventbridge.py
  storage/
    ledger.py
  monitoring/
    events.py
    logging.py
  cli/
    dry_run.py

infra/
  publishing/
    template.yaml or Terraform equivalent

tests/
  publishing/
```

Do not move existing generation logic into `publishing/`; consume its reviewed outputs.

---

# 4. Phase A — AWS authentication hardening

Goal: remove dependence on long-lived GitHub AWS access keys for future publishing deployment.

Tasks:

1. Create a restricted AWS IAM role for GitHub deployment.
2. Configure GitHub Actions OIDC trust.
3. Restrict trust to this repository and approved branch/environment.
4. Grant only deployment permissions needed by the publishing stack.
5. Update future deployment workflow to use OIDC role assumption.
6. Keep the existing render workflow unchanged until separately tested/migrated.
7. Do not grant the GitHub deployment role permission to read the Meta secret value.

Validation:

```text
GitHub can deploy/test approved AWS resources
GitHub cannot read production Meta token
```

Rollback:

- remove OIDC role/trust;
- no effect on existing Instagram generation workflow.

---

# 5. Phase B — Asset finalization

Goal: create immutable Instagram-compatible delivery assets from reviewed output.

Current issue:

```text
renderer output = PNG
Meta documented image publishing path = JPEG
```

Implement a deterministic finalizer, conceptually:

```text
process/instagram_finalize_asset_package.py
```

Responsibilities:

1. read approved review manifest;
2. verify `publish_ready` and review gates;
3. verify expected media count/order;
4. convert reviewed PNG assets to JPEG;
5. verify width/height/aspect policy;
6. calculate SHA-256;
7. create immutable `asset_package_id`;
8. upload to approved S3 prefix;
9. verify uploaded object metadata/hash;
10. write `asset_package.json`.

Recommended path:

```text
s3://eirepolitic-data/instagram/approved/<project>/<period>/<asset_package_id>/media/01.jpg
```

Rules:

- never publish from `latest/`;
- corrections create a new asset package ID;
- canonical S3 objects remain private;
- store order, MIME, dimensions, file size, SHA-256 and optional S3 version ID.

Tests:

- wrong media count;
- missing file;
- PNG→JPEG conversion;
- deterministic ordering;
- hash mismatch;
- overwrite protection;
- invalid MIME/dimensions.

---

# 6. Phase C — Caption templates

Goal: move recurring editorial defaults out of hard-coded Python while preserving exact final caption storage.

Create versioned template files under:

```text
instagram/caption_templates/
```

Each template should distinguish:

```text
required
optional
default/suggested
```

Store final publication caption as:

```yaml
caption:
  text: "complete exact final caption"
  template_ref:
    template_id: ...
    template_version: ...
```

Rules:

- publisher never re-renders template;
- publisher never calls an LLM;
- hashtags/mentions in structured fields must match the exact final text;
- public attribution/disclaimer text must be distinct from internal review notes.

---

# 7. Phase D — Publication ledger

Recommended physical store:

```text
DynamoDB on-demand
```

Logical records:

```text
AssetPackage
PublicationRequest
PublicationApproval
PublicationSchedule
ExecutionAttempt
PublishedMedia
```

Required access patterns:

1. get publication by ID/version;
2. get current publication version;
3. list scheduled publications by date/account;
4. list published media by date/account;
5. load attempts/operations by publication;
6. list `needs_attention`/`auth_blocked`/uncertain states;
7. resolve project/period to eligible asset packages;
8. query recent history for caption-structure reuse.

Use conditional writes/transactions for:

- publication version creation;
- approval immutability;
- execution claim;
- publication-success guard;
- schedule state changes.

Do not finalize a clever single-table design until access patterns are tested against the control-service use cases.

---

# 8. Phase E — Publication fingerprints

Implement canonical serialization and SHA-256 fingerprinting for material approved fields.

Fingerprint should bind to:

```text
platform
account_ref
project/period
publication version
asset package ID
ordered media IDs + hashes
exact caption
alt text
mentions/hashtags
media tags
collaborators
location
first comment
post type
```

Schedule confirmation binds separately to:

```text
scheduled_local
timezone
scheduled_at_utc
```

Rules:

- content/account change → new PublicationRequest version + fresh approval;
- schedule-only change → new schedule revision + explicit time confirmation;
- approved versions are immutable.

Tests:

- same semantic data → same fingerprint;
- reordered media → different fingerprint;
- caption whitespace/newline change → different fingerprint;
- account/tag/location change → different fingerprint.

---

# 9. Phase F — Timezone resolver

Implement deterministic timezone validation using:

```text
Europe/Dublin
```

Store:

```text
scheduled_local
IANA timezone
scheduled_at_utc
```

Rules:

- reject nonexistent DST local times;
- require explicit choice for ambiguous repeated times;
- freeze one UTC instant after approval;
- EventBridge schedule executes the approved UTC instant;
- flexible time window off.

Tests must include Dublin DST forward/backward boundaries.

---

# 10. Phase G — Publication control service

Build deterministic control operations used by High Director.

Required commands:

```text
create draft
edit draft
resolve asset package
validate capabilities
prepare approval summary
record approval
schedule
publish immediately
reschedule
cancel
get scheduled publications
get published publications
get publication details
get failure/status
retry/recover failed publication
```

The service must reject ambiguous mutation commands until the intended publication is resolved.

High Director never writes provider runtime IDs/status directly.

---

# 11. Phase H — Meta capability registry

Implement explicit provider/platform capabilities, for example:

```yaml
platform: instagram
provider: meta_direct
features:
  image: true
  carousel: true
  caption: true
  alt_text: true
  media_tags: true
  collaborators: canary_required
  location: canary_required
  first_comment: canary_required
```

Rules:

- validate before approval;
- unsupported/unverified fields block scheduling;
- never silently drop a requested field;
- canary result can promote `canary_required` to enabled/disabled.

---

# 12. Phase I — Secrets and Meta authentication

After explicit Gate 2 approval:

1. inspect current account subtype/Page linkage;
2. create/configure the Meta App if needed;
3. connect only the required live account/Page;
4. request minimum permissions;
5. record exact token type/scopes/expiry/data-access-expiry properties;
6. store runtime token in AWS Secrets Manager;
7. keep token value out of GitHub/ledger/chat;
8. create auth-health metadata record;
9. implement read-only auth-health check.

Publisher IAM role receives only specific:

```text
secretsmanager:GetSecretValue
```

for the runtime secret.

GitHub deployment role receives no production secret-value permission.

---

# 13. Phase J — Non-publishing Meta canary

Before any visible post:

1. validate account/Page identity;
2. validate current content-publishing permission;
3. generate one temporary S3 retrieval URL;
4. create a non-published Meta media container where supported;
5. confirm Meta successfully retrieves the JPEG asset;
6. inspect container status;
7. do not call `/media_publish`;
8. allow/discard the test container safely.

Also test the current API schema/capability behaviour for:

- alt text;
- media user tags;
- collaborators;
- location.

Record proven capabilities in the registry.

Gate 3:

```text
Do not proceed to visible canary if required capabilities/auth/media retrieval fail.
```

---

# 14. Phase K — EventBridge Scheduler adapter

Implement one EventBridge schedule per approved publication.

Payload:

```json
{
  "publication_id": "pub_...",
  "expected_version": 3
}
```

Rules:

- no caption/media/token in EventBridge payload;
- one-time schedule;
- frozen UTC instant;
- flexible window off;
- automatic deletion after completion if suitable;
- SQS standard DLQ configured;
- bounded retry/event-age policy, not service maximum.

Implement:

```text
create_schedule
get_schedule
update/reschedule
cancel/delete_schedule
verify_schedule
```

Ledger becomes `scheduled` only after verification succeeds.

---

# 15. Phase L — Lambda publisher

Deterministic execution sequence:

```text
receive publication_id/version
→ load immutable approved request
→ verify approval
→ verify schedule/current state
→ atomically acquire execution lease
→ verify asset package hashes/count/order
→ retrieve Meta secret
→ generate temporary media URLs
→ create/reuse child containers
→ poll container readiness
→ create/reuse parent container for carousel
→ /media_publish
→ persist/reconcile Instagram result
→ optional first comment
→ write PublishedMedia
→ release/finalize execution state
```

No LLM inside the worker.

---

# 16. Phase M — Idempotency and recovery

Implement root identity:

```text
publication_id + publication_version
```

Persist operation-level state for:

```text
create_child:<asset>
create_parent
publish_parent
first_comment
```

Critical rules:

- duplicate EventBridge/Lambda invocation must not create duplicate side effects;
- persist every returned Meta container ID immediately;
- after uncertain `/media_publish`, query the same parent container first;
- if container reports `PUBLISHED`, set permanent success guard and never republish;
- secondary first-comment failure never republishes the media;
- auth/permanent-input failures do not auto-retry;
- retry budget stops after an editorially defined lateness window.

Must test crash/replay behaviour before live enablement.

---

# 17. Phase N — Monitoring and operator alerts

Implement structured logs keyed by:

```text
publication_id
publication_version
attempt_id
operation
```

Never log:

- tokens;
- Authorization headers;
- full presigned URLs.

Initial CloudWatch/application alarms:

```text
Scheduler TargetErrorCount >= 1
Scheduler InvocationDroppedCount >= 1
DLQ messages visible >= 1
Lambda Errors >= 1
Lambda Throttles >= 1
PublicationNeedsAttention >= 1
PublicationOutcomeUncertain >= 1
InstagramAuthBlocked >= 1
```

Send initial alerts through SNS/User Notifications/email.

Implement periodic reconciliation for:

- `scheduled` publications vs EventBridge jobs;
- `publishing_unknown` Meta container state;
- published result reconciliation;
- auth-health state.

---

# 18. Phase O — High Director integration

Expose deterministic control actions to High Director rather than direct Meta calls.

High Director should be able to support:

```text
"What's scheduled this week?"
"Show me Tuesday's post."
"Move Tuesday's post to 8pm."
"Cancel Friday's post."
"Use last month's caption structure."
"Why did tonight's post fail?"
"Publish this now."
```

Before any live mutation:

- resolve exact publication;
- show required confirmation;
- invoke deterministic control action;
- report verified ledger state.

Do not report `published` until `PublishedMedia`/provider reconciliation confirms success.

---

# 19. Phase P — Dry-run test suite

Before any visible Meta publication, run automated tests for at least:

## Approval/state

- unapproved publication cannot schedule;
- changed caption invalidates approval;
- changed media/hash invalidates approval;
- wrong account blocks execution;
- schedule-only reschedule retains content approval but requires confirmation.

## Assets

- incomplete carousel;
- out-of-order media;
- hash mismatch;
- missing S3 object;
- unsupported MIME.

## Scheduling

- create/verify/cancel/reschedule;
- duplicate scheduler invocation;
- DST ambiguous/nonexistent times.

## Execution

- child container partial failure;
- Lambda crash after child ID persistence;
- Lambda crash before ID persistence;
- parent creation retry;
- `/media_publish` response timeout;
- Meta status `PUBLISHED` with missing returned media ID;
- auth revoked;
- rate limited;
- first-comment timeout;
- DLQ handling.

## Security

- no Meta token in logs;
- no presigned URL in ledger/log output;
- GitHub role cannot retrieve Meta secret;
- publisher role has minimum required secret/S3/data permissions.

---

# 20. Phase Q — Visible canary publication

This requires a separate explicit approval of the exact test post.

Before publishing, High Director must show:

```text
account
project/period
media count/order
exact caption
hashtags/mentions
media tags
collaborators
location
alt text
first comment
immediate/scheduled mode
local Dublin time + UTC
```

Only after explicit confirmation call `/media_publish`.

Verify:

- correct Instagram account;
- correct carousel order;
- caption exact match;
- alt text;
- media tags;
- collaborators/location if enabled;
- Instagram Media ID;
- permalink;
- ledger records;
- alerts/logs;
- replay safety.

Do not enable general production scheduling automatically after this test.

---

# 21. Phase R — Limited production enablement

After explicit production approval:

Enable only:

```text
one Instagram account
static image + carousel
scheduled + immediate
human approval every publication
cancel/reschedule
status/history
validated metadata capabilities
```

Keep disabled initially:

```text
Reels
Stories
recurring unattended publishing
automatic deletion
published caption editing
multi-platform production
```

Review the first several production executions before expanding scope.

---

# 22. Deployment order

Recommended implementation order:

```text
1. repo schemas/models/tests
2. asset finalizer
3. caption templates
4. DynamoDB ledger
5. fingerprint/timezone/state logic
6. control service
7. GitHub OIDC / AWS roles
8. Secrets Manager references
9. scheduler adapter
10. Meta client/provider adapter
11. Lambda publisher
12. recovery/reconciliation
13. monitoring/alerts
14. High Director control integration
15. dry-run test suite
16. non-publishing Meta canary
17. visible canary after explicit approval
18. limited production after explicit approval
```

This order keeps live-provider risk until near the end.

---

# 23. Decision gates still requiring user approval

The architecture and v1 scope are approved.

Still require explicit approval before:

## Gate 1

Any change to the existing Professional account subtype or Facebook Page linkage.

## Gate 2

Creating/configuring the Meta App and live account authorization/credentials.

## Gate 3

Provisioning/enabling production AWS publishing infrastructure if implementation reaches that stage.

## Gate 4

Calling `/media_publish` for the first visible canary.

## Gate 5

Enabling general production scheduling.

---

# 24. Immediate next implementation step

The safest first implementation work, once implementation itself is approved, is **repo-only and non-live**:

```text
A. define publication schemas/models
B. implement fingerprint/state validation
C. implement asset finalizer + tests
D. implement caption-template loading + tests
```

This can be completed without touching:

- the Instagram account;
- Meta credentials;
- EventBridge;
- Lambda;
- production AWS resources.

After that foundation is reviewed, proceed to the AWS/control-plane layers.
