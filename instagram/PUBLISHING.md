# Instagram publishing quick start

## Before scheduling anything

Run **Instagram schedule inventory** (`.github/workflows/instagram_schedule_inventory.yml`).

This read-only workflow is the standard calendar check. It lists future publications from the canonical DynamoDB ledger (`eirepolitic-publications`, `state=scheduled`) in chronological order and verifies that each one has the matching live EventBridge Scheduler job in group `eirepolitic-instagram`.

Use that inventory before choosing a new scheduled time so you can see what is already planned. If a row reports a missing/mismatched Scheduler job, resolve that operational issue before adding another schedule.

DynamoDB is the canonical schedule record. EventBridge is the live execution mirror and its one-time jobs delete themselves after execution.

## Normal human/operator path

1. Run **Instagram factory render (generic)**.
2. Review the preview and approve the exact output.
3. Copy the factory GitHub Actions run ID.
4. If scheduling, run **Instagram schedule inventory** and review the upcoming calendar.
5. Run **Instagram publish (standard)**.
6. Enter:
   - `factory_run_id`
   - `approved_by`
   - `mode` = `scheduled` or `immediate`
   - `scheduled_local` + `timezone` only when scheduled
7. Leave `options_json={}` unless tags/collaborators/location/first comment/alt text are explicitly required.
8. Use `caption_override` only if the factory artifact has no caption.

## Agent/High Director path

If the GitHub integration cannot pass workflow inputs, set these repository Actions variables and dispatch the **same** `instagram_publish_standard.yml` workflow:

- `INSTAGRAM_FACTORY_RUN_ID`
- `INSTAGRAM_APPROVED_BY`
- `INSTAGRAM_PUBLISH_MODE` = `scheduled` or `immediate`
- `INSTAGRAM_SCHEDULED_LOCAL` for scheduled mode
- `INSTAGRAM_TIMEZONE` (normally `America/Vancouver` when Pacific time was requested)
- `INSTAGRAM_CAPTION_OVERRIDE` only when needed
- `INSTAGRAM_OPTIONS_JSON` (normally `{}`)

Before setting a scheduled time, dispatch `instagram_schedule_inventory.yml` and inspect its workflow summary. Do not create a separate agent-only calendar or infer availability from old conversations.

For safety, leave the normal maintenance selector `HIGH_DIRECTOR_AWS_OPERATION` at `infrastructure-status` except while performing an explicit maintenance operation.

## Hard rules

Do not upload assets manually, call Meta directly, build one-off EventBridge schedules, or add temporary Lambda publish actions.

The canonical implementation/runbook is:

`docs/operations/instagram_publishing_standard.md`

The factory remains non-publishing by design (`publication_enabled=false`). The separate standard publisher creates the exact approval fingerprint and executes only approved ledger records.
