# Oireachtas final cutover request package

**Status:** request package, not approval  
**Last updated:** 2026-06-16  
**No cutover is approved by this document.**

## Current recommendation

Do not repoint downstream consumers unless the user explicitly approves the exact consumer-specific change. The first Instagram consumer smoke test has passed, but row-count gaps remain because current unified latest outputs are still sample-sized.

## Evidence completed

| Area | Evidence |
|---|---|
| Core tables | T01-T23, G01-G05, C01-C03 validated with DQ pass on sample/test runs. |
| Latest publishing safety | P01 run `27431598142`: `mode=test` suppresses latest pointer writes. |
| Dynamic schedules | P02 updated weekly/monthly/yearly scheduled date windows. |
| Compatibility adapters | P03 run `27431601110`: roster and vote compat outputs written to unified compat paths. |
| Member profile trial | P04 run `27432417013`: trial metrics written to unified compat paths, DQ pass. |
| Adapter comparison | P05 run `27432358137`: legacy-vs-compat comparison report, DQ pass. |
| Production checklist | P06 checklist added in `docs/oireachtas_production_readiness_checklist.md`. |
| Consumer smoke plan | P07 plan added in `docs/oireachtas_consumer_smoke_test_plan.md`. |
| Consumer smoke workflow | P08 workflow added in `.github/workflows/oireachtas_instagram_consumer_smoke.yml`. |
| Consumer smoke validation | P08 run `27636367782`: Instagram renderer completed with compat members key and uploaded artifact `oireachtas-instagram-consumer-smoke-output`. |

## Exact trial keys

```text
processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv
processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv
processed/oireachtas_unified/compat/members/parquets/member_profile_metrics_2025_trial.parquet
```

## Known row-count gaps

Current unified compat outputs are based on limited latest/test outputs, not full production-sized runs.

| Dataset | Legacy rows | Compat/trial rows | Note |
|---|---:|---:|---|
| roster | 176 | 10 | Compat output currently follows limited `gold_current_members` latest. |
| member votes | 30,968 | 512 | Compat output currently follows limited `silver_member_votes` latest. |
| member profile metrics | 174 | 10 | Trial output follows limited compat roster. |

These gaps are expected until a production-sized unified run refreshes the latest pointers.

## Consumer smoke result

The first Instagram consumer smoke test used:

```bash
INSTAGRAM_MEMBERS_DATASET_KEYS=processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
```

Run result:

```text
Workflow ID: 297114820
Run ID: 27636367782
Run number: 1
Status: success
Artifact: oireachtas-instagram-consumer-smoke-output
Artifact ID: 7674904547
```

The workflow rendered `Wicklow-Wexford` / `Brian Brennan` into `generated_posts_oireachtas_compat_trial/` and validated that the compat members S3 key was used.

## Consumer-specific change that would require approval

For Instagram constituency rendering, the smallest reversible change is to set this environment variable in the target workflow only:

```bash
INSTAGRAM_MEMBERS_DATASET_KEYS=processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
```

Keep these enrichment keys unchanged unless separate replacements are built and validated:

```text
processed/members/members_summaries.csv
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
processed/debates/debate_speeches_classified.csv
processed/constituencies/constituency_images.csv
```

For member profile metrics, the reversible trial cutover would use:

```bash
MEMBERS_INPUT_KEY=processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
MEMBER_VOTES_INPUT_KEY=processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv
```

Do not change production output keys until explicitly approved.

## Approval required before applying any cutover

Use a consumer-specific approval phrase:

```text
Approved: cut over <consumer name> from legacy Oireachtas keys to unified compatibility outputs.
```

Examples:

```text
Approved: cut over Instagram constituency renderer from legacy Oireachtas keys to unified compatibility outputs.
Approved: cut over member profile metrics from legacy Oireachtas keys to unified compatibility outputs.
```

## Rollback

Rollback is environment/config-only.

Instagram rollback:

```bash
unset INSTAGRAM_MEMBERS_DATASET_KEYS
```

Member profile metrics rollback:

```bash
MEMBERS_INPUT_KEY=raw/members/oireachtas_members_34th_dail.csv
MEMBER_VOTES_INPUT_KEY=processed/votes/dail_vote_member_records.csv
MEMBER_PROFILE_METRICS_OUTPUT_CSV_KEY=processed/members/member_profile_metrics_2025.csv
MEMBER_PROFILE_METRICS_OUTPUT_PARQUET_KEY=processed/members/parquets/member_profile_metrics_2025.parquet
```

## Stop point

Stop here unless explicit approval is provided. The next safe action is to run a production-sized unified refresh, then rerun adapter comparison and smoke tests before any cutover.
