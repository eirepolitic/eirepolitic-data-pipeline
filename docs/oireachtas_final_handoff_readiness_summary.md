# Oireachtas unified pipeline final handoff and readiness summary

**Status:** cutover validated; scheduled monthly follow-up required  
**Last updated:** 2026-07-04

## Executive summary

The unified Oireachtas pipeline now has validated silver, gold, control, compatibility, and enrichment outputs. Controlled pre-production cutovers have been applied and validated for:

```text
Instagram constituency renderer
Instagram campaign renderer default spec
Member profile metrics
Member photo URLs
Constituency images
Member summaries
Classified speech issue labels
```

Final post-cutover comparison and mismatch review passed. The only known data caveat remains the two legacy-only roster members already documented.

Scheduled automation is not fully green because the latest monthly scheduled refresh failed on `2026-07-01`. Weekly has a successful manual post-patch validation but still needs observation on the next scheduled run.

## Current production/default workflow inputs

### Member profile metrics

Workflow:

```text
.github/workflows/build_member_profile_metrics_2025.yml
```

Validated run:

```text
266755732 / 28684033733 / success
```

Default inputs:

```yaml
MEMBERS_INPUT_KEY: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
MEMBER_VOTES_INPUT_KEY: "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv"
MEMBER_PHOTOS_INPUT_KEY: "processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv"
DEBATE_ISSUES_INPUT_KEY: "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv"
```

### Instagram constituency renderer

Workflow:

```text
.github/workflows/instagram_constituency_test.yml
```

Validated run:

```text
261945698 / 28672901108 / success / artifact 8071309560
```

Default inputs:

```yaml
INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
INSTAGRAM_MEMBER_SUMMARIES_DATASET_KEYS: "processed/oireachtas_unified/compat/text/members_summaries_compat.csv"
INSTAGRAM_CONSTITUENCY_IMAGES_DATASET_KEYS: "processed/oireachtas_unified/compat/media/constituency_images_compat.csv"
```

### Instagram campaign renderer

Workflow:

```text
.github/workflows/instagram_campaign_render.yml
```

Validated run:

```text
271160957 / 28415050102 / success / artifact 7969146127
```

Default state:

```text
spec_file=render_spec.yml
upload_preview=false
```

## Enrichment compatibility outputs

### Classified speech issue labels

Validated full run:

```text
304470256 / 28683964925 / success / artifact 8074954501
```

Output:

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

Full-row evidence:

```text
source_rows: 47275
output_rows: 47275
compat_rows: 47275
DQ: pass
```

### Member photo URLs

Validated run:

```text
304478490 / 28422342745 / success / artifact 7971687268
```

Output:

```text
processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv
```

Evidence:

```text
source_rows: 174
compat_rows: 174
photo_url_populated_count: 174
DQ: pass
```

### Constituency images

Validated run:

```text
305600627 / 28547829924 / success / artifact 8022576293
```

Output:

```text
processed/oireachtas_unified/compat/media/constituency_images_compat.csv
```

Evidence:

```text
source_rows: 43
compat_rows: 43
image_locator_missing_count: 0
DQ: pass
```

### Member summaries

Validated run:

```text
306762190 / 28672859337 / success / artifact 8071290965
```

Output:

```text
processed/oireachtas_unified/compat/text/members_summaries_compat.csv
```

Evidence:

```text
source_rows: 174
compat_rows: 174
summary_text_missing_count: 0
DQ: pass
```

## Final validation sweep

Compatibility comparison:

```text
294874693 / 28691936308 / success / artifact 8077413255
```

Mismatch review:

```text
297343766 / 28691938402 / success / artifact 8077412883
```

Known roster caveat:

```text
Catherine Connolly — Independent — Galway West
Paschal Donohoe — Fine Gael — Dublin Central
```

These are legacy-only roster records. Current member-profile metrics remain aligned at 174 matched members.

## Scheduled refresh status

Weekly:

```text
Latest run: 28421557467
Event: workflow_dispatch
Result: success
```

Monthly:

```text
Latest run: 28504651002
Event: schedule
Result: failure
Failed step: Run monthly table set
Artifact ID: 8004493556
```

Yearly:

```text
Latest run: 27397123885
Event: workflow_dispatch
Result: success
```

## Rollback paths

### Member profile metrics rollback

Remove these env vars from `.github/workflows/build_member_profile_metrics_2025.yml` as needed:

```yaml
MEMBERS_INPUT_KEY
MEMBER_VOTES_INPUT_KEY
MEMBER_PHOTOS_INPUT_KEY
DEBATE_ISSUES_INPUT_KEY
```

The script defaults fall back to legacy source paths.

### Instagram constituency rollback

Remove these env vars from `.github/workflows/instagram_constituency_test.yml` as needed:

```yaml
INSTAGRAM_MEMBERS_DATASET_KEYS
INSTAGRAM_MEMBER_SUMMARIES_DATASET_KEYS
INSTAGRAM_CONSTITUENCY_IMAGES_DATASET_KEYS
```

### Enrichment rollback

Legacy keys remain preserved:

```text
processed/debates/debate_speeches_classified.csv
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
processed/constituencies/constituency_images.csv
processed/members/members_summaries.csv
```

## Remaining work

1. Investigate monthly scheduled refresh run `28504651002`.
2. Observe the next scheduled weekly refresh after the debate-record DQ patch.
3. Consider adding a single orchestration workflow that runs refresh, adapters, comparison, mismatch review, and consumer validations in order.
4. Decide whether legacy enrichment workflows should remain active, be renamed as legacy, or be disabled after a longer observation window.

## Final readiness statement

The unified data model and downstream consumer compatibility cutovers are validated for controlled pre-production use. The final blocker for declaring scheduled production readiness is the monthly scheduled refresh failure investigation.
