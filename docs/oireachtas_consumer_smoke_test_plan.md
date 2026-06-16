# Oireachtas consumer smoke test plan

**Status:** trial plan  
**Last updated:** 2026-06-12  
**No production consumer defaults are changed by this plan.**

## Safest first consumer test

Use the existing Instagram constituency local HTML renderer because it is visual, artifact-only, and already reads S3 datasets without publishing.

The trial should override only the member roster input to use the unified compatibility roster:

```text
processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
```

Keep the following enrichment inputs on legacy keys for now because they are outside the deterministic unified Oireachtas model:

```text
processed/members/members_summaries.csv
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
processed/debates/debate_speeches_classified.csv
processed/constituencies/constituency_images.csv
```

## Trial constituency

Use `Wicklow-Wexford` for the first smoke test because it appears in the current limited unified current-members sample.

Current sampled member:

```text
Brian Brennan — Fine Gael — Wicklow-Wexford
```

## Success criteria

| Check | Expected result |
|---|---|
| Render command completes | pass |
| `post_context.json` exists | pass |
| HTML slides exist | pass |
| PNG slides exist | pass unless screenshot rendering fails |
| `datasets_used` shows compat roster key | pass |
| Output is uploaded only as a GitHub Actions artifact | pass |
| No S3 production output is overwritten | pass |
| No Instagram publishing occurs | pass |

## Trial constraints

- Do not alter `instagram/specs/constituency_test_post.yml` defaults.
- Do not alter the existing production/manual Instagram workflows.
- Do not change `DATASET_CANDIDATES` default legacy paths without environment overrides.
- Do not disable legacy workflows.
- Do not promote trial output to production.

## Follow-up after trial

If the smoke test passes, prepare a cutover request package listing the exact change needed for approval. If it fails, keep the current legacy inputs and fix the trial path only.
