# Oireachtas approved cutover patch plan

**Status:** patch plan only  
**Last updated:** 2026-06-17  
**No patch in this document has been applied.**

This document describes the smallest reversible patches to apply after explicit approval.

## Approval required

Do not apply any patch below unless the user provides a consumer-specific approval phrase:

```text
Approved: cut over <consumer name> from legacy Oireachtas keys to unified compatibility outputs.
```

## Instagram constituency renderer patch

Target workflow:

```text
.github/workflows/instagram_constituency_test.yml
```

Add this environment variable under `jobs.render_post.env`:

```yaml
      INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
```

No spec file changes are required because `process/instagram_render_post.py` already supports dataset overrides.

Rollback:

```yaml
      # Remove INSTAGRAM_MEMBERS_DATASET_KEYS to restore the default legacy member roster.
```

## Member profile metrics patch

Target workflow:

```text
.github/workflows/build_member_profile_metrics_2025.yml
```

A cautious first cutover would switch only the input keys while keeping production output keys unchanged:

```yaml
      MEMBERS_INPUT_KEY: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
      MEMBER_VOTES_INPUT_KEY: "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv"
```

Do not change these production output keys unless separately approved:

```text
processed/members/member_profile_metrics_2025.csv
processed/members/parquets/member_profile_metrics_2025.parquet
```

Rollback:

```yaml
      MEMBERS_INPUT_KEY: "raw/members/oireachtas_members_34th_dail.csv"
      MEMBER_VOTES_INPUT_KEY: "processed/votes/dail_vote_member_records.csv"
```

## Safer staged option

Before changing production workflows, create consumer-specific trial workflows or add manual-only workflow inputs that default to legacy keys. This keeps scheduled/manual production behavior unchanged unless an operator deliberately selects unified compatibility keys.

## Post-cutover validation

After any approved cutover patch:

1. Run the patched consumer workflow manually.
2. Confirm generated outputs exist and use the unified compatibility key.
3. Compare output row counts and rendered artifacts against the last legacy run.
4. Keep legacy workflows and keys available for one complete scheduled cycle.
5. Roll back immediately if row counts, rendered output, or generated content regresses.

## Current stop point

This plan is ready, but no cutover patch should be applied until P13 mismatch review is complete and explicit approval is received.
