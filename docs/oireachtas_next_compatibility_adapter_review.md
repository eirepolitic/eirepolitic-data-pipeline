# Oireachtas next compatibility adapter review

**Status:** no new adapter required  
**Last updated:** 2026-06-30

## Reviewed next consumer

The next consumer reviewed was:

```text
Instagram Campaign Render (Manual)
```

Workflow:

```text
.github/workflows/instagram_campaign_render.yml
```

Production spec:

```text
instagram/campaigns/member_profile_batch_v1/render_spec.yml
```

Data source:

```text
s3://eirepolitic-data/processed/members/member_profile_metrics_2025.csv
```

## Adapter decision

No new compatibility adapter is required for this consumer.

Reason:

- The campaign renderer already expects the legacy-shaped member profile metrics table.
- The member profile metrics workflow has been cut over to build that legacy-shaped output from unified compatibility inputs.
- Adding another adapter would duplicate the same shape and add unnecessary maintenance.

## Current dependency chain

```text
processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv
        ↓
.github/workflows/build_member_profile_metrics_2025.yml
        ↓
processed/members/member_profile_metrics_2025.csv
        ↓
instagram/campaigns/member_profile_batch_v1/render_spec.yml
        ↓
.github/workflows/instagram_campaign_render.yml
```

## Future adapter trigger

Add a new adapter only if a downstream consumer needs a legacy-shaped table that is not already produced by one of the existing compatibility or metrics outputs.

Candidate future adapter examples:

```text
classified debate issue summaries
member photo URL index
constituency image index
member summary/background index
```

These are not deterministic Oireachtas table replacements yet, so they should stay on the current legacy/enrichment paths until separate enrichment models are built.
