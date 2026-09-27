# Instagram publishing standard

## Canonical rule

Production Instagram publishing uses one supported path:

1. run `.github/workflows/instagram_factory_render.yml`;
2. visually review the generated preview;
3. note the factory GitHub Actions run ID;
4. **before choosing a scheduled time, run `.github/workflows/instagram_schedule_inventory.yml` and review all upcoming scheduled posts**;
5. run `.github/workflows/instagram_publish_standard.yml` with that run ID;
6. choose `scheduled` or `immediate`;
7. for scheduled mode, supply a local timestamp and IANA timezone;
8. let the workflow promote the reviewed artifact, approve the exact fingerprint, and publish/schedule it.

Agents must not rebuild this flow manually with ad-hoc S3 uploads, direct Meta calls, custom EventBridge schedules, or one-off Lambda actions.

## Where scheduled posts are kept

There are two related stores, with different roles:

### Canonical schedule ledger — DynamoDB

The authoritative record of scheduled Instagram publications is the DynamoDB table:

`eirepolitic-publications`

Scheduled control records have:

- `state = scheduled`;
- `scheduled_local`;
- `timezone`;
- `scheduled_at_utc`;
- publication ID/version;
- project/period identity;
- approval and immutable asset references.

The table has the `state-scheduled_at-index` index, which is the canonical way to list upcoming scheduled posts in chronological order.

This ledger remains authoritative even after a one-time EventBridge schedule deletes itself following execution.

### Live execution mirror — EventBridge Scheduler

The live one-time execution jobs are in EventBridge Scheduler group:

`eirepolitic-instagram`

These jobs are an execution mirror of the DynamoDB schedule record. They are not the long-term calendar/source of truth because one-time schedules use `ActionAfterCompletion=DELETE`.

A valid upcoming scheduled publication should normally have:

- a DynamoDB control record in `scheduled` state; and
- a matching enabled EventBridge schedule with the expected publication ID/version payload, target Lambda, execution role, and DLQ.

The schedule inventory workflow cross-checks these two layers and flags a missing/mismatched EventBridge job.

## Mandatory pre-scheduling check

Before an agent selects a time for a new scheduled post, it must run:

**Instagram schedule inventory**  
`.github/workflows/instagram_schedule_inventory.yml`

This is a read-only workflow. It queries upcoming `scheduled` records from DynamoDB in chronological order and verifies the matching EventBridge schedule for each one.

The workflow summary shows, for every upcoming post:

- requested local publish time;
- timezone;
- project ID;
- publication ID;
- live Scheduler state;
- whether the EventBridge schedule matches the ledger record.

If the inventory is empty, there are no upcoming scheduled Instagram publications in the canonical ledger.

If the inventory shows a missing or mismatched EventBridge schedule, treat that as an operational issue to resolve before adding another schedule.

The agent should use this inventory to choose a new time that does not unintentionally collide with existing scheduled content. The repository does not currently impose an automatic minimum spacing rule; spacing/editorial cadence remains an operator/content decision informed by the inventory.

## Inputs agents should use

Required for all publishes:

- `factory_run_id`: run ID from `Instagram factory render (generic)`;
- `approved_by`: identity of the human/operator who reviewed and approved the exact factory output;
- `mode`: `scheduled` or `immediate`.

Scheduled mode additionally requires:

- `scheduled_local`: `YYYY-MM-DDTHH:MM:SS`;
- `timezone`: IANA timezone, normally `America/Vancouver` for Pacific scheduling.

Optional inputs:

- `caption_override`: use only when the factory artifact has no caption file;
- `options_json`: advanced Instagram fields. Default `{}` means no media tags, collaborators, location, first comment, or alt-text override.

Example `options_json`:

```json
{
  "alt_text": ["Alt text for slide 1", "Alt text for slide 2"],
  "media_tags": [],
  "collaborators": [],
  "location_id": null,
  "first_comment": null
}
```

## Factory handoff contract

Successful generic factory renders write `publication_handoff.json` into the uploaded `generated_render/` artifact. The handoff uses only paths relative to the artifact root and records:

- schema version;
- project ID and period;
- QA result;
- factory review/publication safety flags;
- ordered slide paths and dimensions;
- caption path when one exists;
- source manifest path;
- the explicit rule that factory rendering itself never publishes.

The factory continues to enforce `publication_enabled=false`. Publishing authority exists only in the separate publication ledger approval created by the standard publishing workflow.

## What the publishing workflow does

The workflow:

1. downloads the exact factory artifact by run ID;
2. locates `publication_handoff.json`;
3. validates factory QA and safety flags;
4. converts every slide to deterministic delivery JPEG;
5. uploads immutable content-addressed assets to the private approved-assets S3 bucket;
6. stores the immutable `AssetPackage` in DynamoDB;
7. creates a deterministic `PublicationRequest`;
8. parses hashtags and caption mentions from the exact caption;
9. records an approval fingerprint tied to exact caption/options/assets;
10. either invokes the publisher Lambda immediately or creates and verifies a one-time EventBridge Scheduler job.

Scheduled jobs contain only publication identity/version. Approved content stays in DynamoDB/S3. Credentials stay in Secrets Manager.

## Runtime behavior

The production Lambda accepts three classes of request:

- `{"action":"healthcheck"}` — read-only Meta connection check;
- `{"action":"execute_publication", ...}` — immediate execution of an already-approved publication;
- EventBridge Scheduler payload `{"publication_id":"...","expected_version":N}` — scheduled execution.

The Lambda does not accept raw caption/image content from callers. It reloads the approved request, immutable asset package, approval fingerprint, secret, and durable execution state from AWS before contacting Meta.

Durable execution state prevents duplicate media creation/publication across retries. Meta container IDs and permanent media IDs are persisted. EventBridge one-time schedules use `ActionAfterCompletion=DELETE` and a DLQ.

## Proven production evidence

The system was proven in production on September 26, 2026:

- Gate 4 immediate canary: real image post published successfully through Meta `/media` and `/media_publish`, then manually deleted;
- Gate 5 scheduled canary: one-time EventBridge Scheduler invocation successfully published the test post at the requested Pacific time, then it was manually deleted.

Those canaries validated the real Instagram Professional account, Meta credentials, S3 delivery path, DynamoDB state, Lambda execution, durable idempotency, EventBridge Scheduler role/DLQ path, and permanent media-ID recording.

## Agent operating procedure

When asked to publish a factory-generated post:

1. confirm the factory run has completed and its preview has been reviewed;
2. obtain the factory run ID;
3. do not edit generated assets during the publishing step;
4. if scheduling, run `Instagram schedule inventory` and review existing upcoming posts before proposing/choosing a time;
5. trigger `Instagram publish (standard)`;
6. use `scheduled` unless the user explicitly asks for immediate publishing;
7. use the user's requested timezone/time exactly;
8. report the resolved local and UTC scheduled time;
9. after the workflow succeeds, report the publication ID and schedule state;
10. do not create parallel custom scheduling/publishing mechanisms.

If the factory artifact has no caption, stop and obtain/provide the exact approved caption via `caption_override`. Do not invent missing publication copy unless the user has explicitly asked the agent to write it.

## Maintenance workflow

`.github/workflows/deploy_instagram_publisher_lambda.yml` is infrastructure/runtime maintenance only. It is not the normal posting interface.

Supported maintenance operations are Lambda deployment, the three CloudFormation stacks, healthcheck, and infrastructure status.

## Canonical implementation files

- `.github/workflows/instagram_factory_render.yml`
- `.github/workflows/instagram_schedule_inventory.yml`
- `.github/workflows/instagram_publish_standard.yml`
- `.github/workflows/deploy_instagram_publisher_lambda.yml`
- `instagram/factory/publication_handoff.py`
- `publishing/schedule_inventory.py`
- `publishing/standard_pipeline.py`
- `publishing/lambda_handler.py`
- `publishing/aws_runtime.py`
- `publishing/scheduler.py`
- `publishing/dynamodb_runtime.py`
- `infra/publishing/*.yml`

This document is the canonical operating contract. Other repositories should link here rather than duplicate implementation details.
