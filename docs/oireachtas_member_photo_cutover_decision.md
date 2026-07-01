# Oireachtas member photo production cutover decision

**Status:** approved for controlled pre-production cutover  
**Last updated:** 2026-06-30

## Decision

The member photo compatibility output is ready for controlled pre-production use by member-profile metrics.

Production public publishing is not changed by this decision.

## Candidate input

```text
processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv
```

## Current legacy photo inputs

The member-profile metrics script currently supports these photo input candidates:

```text
processed/members/members_photo_urls.csv
processed/members/member_photos/members_photo_urls.csv
```

## Evidence

### Member photo enrichment trial

```text
Workflow ID: 304478490
Run ID: 28422342745
Result: success
DQ: pass
```

Review result:

```text
source_rows: 174
output_rows: 174
compat_rows: 174
photo_url_populated_count: 174
photo_url_missing_count: 0
```

### Member-profile consumer trial

```text
Workflow ID: 294874303
Run ID: 28461617338
Result: success
DQ: pass
```

Review result:

```text
legacy_rows: 174
trial_rows: 174
matched_member_count: 174
trial_only_member_count: 0
legacy_only_member_count: 0
common_column_count: 12
```

## Recommended production workflow patch

Patch:

```text
.github/workflows/build_member_profile_metrics_2025.yml
```

Add:

```yaml
      MEMBER_PHOTOS_INPUT_KEY: "processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv"
```

## Rollback

Remove the `MEMBER_PHOTOS_INPUT_KEY` override. The script will fall back to the legacy photo candidate keys.

## Risk assessment

Low risk:

- The compat file has 174 rows.
- All 174 rows have populated photo URLs.
- Member-profile trial matched all 174 member codes.
- No legacy photo URL key is overwritten.

## Current recommendation

Apply the pre-production workflow patch when ready, then run `Build Member Profile Metrics 2025 (Manual)` once to validate production workflow behavior.
