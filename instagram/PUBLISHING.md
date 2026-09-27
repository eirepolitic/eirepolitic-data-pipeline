# Instagram publishing quick start

## Normal human/operator path

1. Run **Instagram factory render (generic)**.
2. Review the preview and approve the exact output.
3. Copy the factory GitHub Actions run ID.
4. Run **Instagram publish (standard)**.
5. Enter:
   - `factory_run_id`
   - `approved_by`
   - `mode` = `scheduled` or `immediate`
   - `scheduled_local` + `timezone` only when scheduled
6. Leave `options_json={}` unless tags/collaborators/location/first comment/alt text are explicitly required.
7. Use `caption_override` only if the factory artifact has no caption.

## Agent/High Director path

If the GitHub integration cannot pass workflow inputs, set these repository Actions variables and dispatch the **same** `instagram_publish_standard.yml` workflow:

- `INSTAGRAM_FACTORY_RUN_ID`
- `INSTAGRAM_APPROVED_BY`
- `INSTAGRAM_PUBLISH_MODE` = `scheduled` or `immediate`
- `INSTAGRAM_SCHEDULED_LOCAL` for scheduled mode
- `INSTAGRAM_TIMEZONE` (normally `America/Vancouver` when Pacific time was requested)
- `INSTAGRAM_CAPTION_OVERRIDE` only when needed
- `INSTAGRAM_OPTIONS_JSON` (normally `{}`)

After dispatch, monitor that workflow. Do not create a separate agent-only publishing mechanism.

For safety, leave the normal maintenance selector `HIGH_DIRECTOR_AWS_OPERATION` at `infrastructure-status` except while performing an explicit maintenance operation.

## Hard rules

Do not upload assets manually, call Meta directly, build one-off EventBridge schedules, or add temporary Lambda publish actions.

The canonical implementation/runbook is:

`docs/operations/instagram_publishing_standard.md`

The factory remains non-publishing by design (`publication_enabled=false`). The separate standard publisher creates the exact approval fingerprint and executes only approved ledger records.
