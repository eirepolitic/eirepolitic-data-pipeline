# Instagram publishing quick start

For agents/operators, use exactly this sequence:

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

Do not upload assets manually, call Meta directly, build one-off EventBridge schedules, or add temporary Lambda publish actions.

The canonical implementation/runbook is:

`docs/operations/instagram_publishing_standard.md`

The factory remains non-publishing by design (`publication_enabled=false`). The separate standard publisher creates the exact approval fingerprint and executes only approved ledger records.
