# Oireachtas production readiness checklist

**Status:** approval checklist  
**Last updated:** 2026-06-12  
**No downstream cutover is approved by this document.**

Use this checklist before repointing Instagram, member profile metrics, or any other downstream consumer to unified Oireachtas outputs.

## Approval gate

Do not repoint consumers or disable legacy workflows until all required checks below are complete and the user explicitly approves the change.

Required approval phrase to record in a future commit or ticket:

```text
Approved: cut over <consumer name> from legacy Oireachtas keys to unified compatibility outputs.
```

## Required checks

| Area | Check | Status |
|---|---|---|
| Table validation | All silver/gold/control tables required by the consumer have DQ pass. | done for tested samples |
| Registry | Required tables are marked `confirmed` in `configs/oireachtas/tables.yml`. | done |
| Latest publishing | `mode=test` cannot overwrite `processed/oireachtas_unified/latest/*`. | done |
| Scheduled windows | Weekly/monthly/yearly scheduled workflows use dynamic date windows. | done |
| Compatibility outputs | Required compat CSVs exist under `processed/oireachtas_unified/compat/...`. | done for roster and member votes |
| Side-by-side trial | Member profile metric trial writes to non-legacy output keys. | done, run `27432417013` |
| Adapter comparison | Legacy inputs are compared to compatibility outputs. | done, run `27432358137` |
| Consumer smoke test | Target downstream workflow runs using trial/compat keys. | pending |
| Rollback | Consumer can be switched back to legacy environment variables or config. | documented |
| User approval | Explicit approval exists for each consumer cutover. | pending |

## Current safe outputs

These outputs are safe to inspect and trial because they do not overwrite legacy keys:

```text
processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv
processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv
processed/oireachtas_unified/compat/members/parquets/member_profile_metrics_2025_trial.parquet
```

## Current legacy keys to keep active

Keep these in place until a consumer-specific cutover is approved:

```text
raw/members/oireachtas_members_34th_dail.csv
processed/votes/dail_vote_member_records.csv
processed/debates/debate_speeches_classified.csv
processed/members/member_profile_metrics_2025.csv
processed/members/parquets/member_profile_metrics_2025.parquet
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
processed/constituencies/constituency_images.csv
```

## Latest validation evidence

| Check | Run | Result | Notes |
|---|---:|---|---|
| Member profile trial | `27432417013` | success / DQ pass | Trial rows: 10; matched legacy member codes: 10; output stays under unified compat paths. |
| Compatibility adapter comparison | `27432358137` | success / DQ pass | Roster compat: 10 of 176 legacy members matched; vote compat: 172 matched vote member codes; row gaps are expected until production-sized unified latest outputs exist. |

Review outputs:

```text
review/member_profile_metrics_trial/latest/{manifest.json,sample.csv,dq.json,report.md}
review/compat_adapter_comparison/latest/{manifest.json,sample.csv,dq.json,report.md}
```

## Recommended cutover order

1. Run a consumer smoke test with trial/compat keys.
2. Review row counts, missing keys, and rendered output.
3. Approve one consumer at a time.
4. Repoint that consumer through config/environment variables.
5. Keep legacy workflow active for at least one complete scheduled cycle.
6. Roll back immediately if consumer output regresses.

## Rollback

Rollback should be config-only if the trial approach is followed.

For member profile metrics:

```bash
MEMBERS_INPUT_KEY=raw/members/oireachtas_members_34th_dail.csv
MEMBER_VOTES_INPUT_KEY=processed/votes/dail_vote_member_records.csv
MEMBER_PROFILE_METRICS_OUTPUT_CSV_KEY=processed/members/member_profile_metrics_2025.csv
MEMBER_PROFILE_METRICS_OUTPUT_PARQUET_KEY=processed/members/parquets/member_profile_metrics_2025.parquet
```

For Instagram rendering, restore the old S3 source keys in the workflow/spec/environment and rerun the same preview workflow.

## Known caveats before approval

- Current unified latest roster output was produced from limited validation data, so compat roster currently has fewer rows than the old legacy roster.
- Deterministic unified outputs do not replace LLM/classified issue data such as `processed/debates/debate_speeches_classified.csv`.
- Photo URLs and constituency image indexes remain outside the deterministic Oireachtas model.
- Test mode is now safer, but production-like incremental/full runs still need row-count review before consumer cutover.

## Decision

Current recommendation: **do not cut over yet**. Run a consumer smoke test using trial keys before requesting approval.
