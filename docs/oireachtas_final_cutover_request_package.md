# Oireachtas final cutover request package

**Status:** pre-production cutover applied  
**Last updated:** 2026-06-30

## Current recommendation

The first two consumer cutovers have been applied because these systems are not yet in use. Continue monitoring and keep rollback instructions available.

## Evidence completed

| Area | Evidence |
|---|---|
| Core tables | T01-T23, G01-G05, C01-C03 validated with DQ pass. |
| Latest publishing safety | P01 run `27431598142`: `mode=test` suppresses latest pointer writes. |
| Dynamic schedules | P02 updated weekly/monthly/yearly scheduled date windows. |
| Initial compatibility adapters | P03 run `27431601110`: roster and vote compat outputs written under unified compat paths. |
| Initial member profile trial | P04 run `27432417013`: trial metrics written under unified compat paths, DQ pass. |
| Initial adapter comparison | P05 run `27432358137`: legacy-vs-compat comparison report, DQ pass. |
| Production checklist | P06 checklist added in `docs/oireachtas_production_readiness_checklist.md`. |
| Consumer smoke plan | P07 plan added in `docs/oireachtas_consumer_smoke_test_plan.md`. |
| Initial consumer smoke | P08 run `27636367782`: Instagram renderer completed with compat members key. |
| Production-sized refresh plan | P10 plan added in `docs/oireachtas_production_sized_refresh_plan.md`. |
| Production-sized refresh dry run | P11 run `27661934424`: success. `gold_current_members` refreshed to 174 rows; `silver_member_votes` refreshed to 29,805 rows. |
| Post-refresh adapter rerun | P12 adapter run `27661982505`: success; compat roster 174 rows; compat member votes 29,805 rows. |
| Post-refresh member profile trial | P12 trial run `27661985049`: success; trial profile metrics 174 rows; 172 matched legacy member codes. |
| Post-refresh adapter comparison | P12 comparison run `27661990358`: success; roster 174/176 matched, vote member-code coverage 173 matched and 0 legacy-only member codes. |
| Post-refresh Instagram smoke | P12 smoke run `27661992188`: success; artifact `oireachtas-instagram-consumer-smoke-output`, artifact ID `7684743075`. |
| Remaining mismatch review | P13 run `27662884471`: success; mismatch report published under `review/member_code_mismatch_review/latest/`. |
| Cutover patch preparation | P15 document `docs/oireachtas_approved_cutover_patch_plan.md`. |
| Instagram cutover | P17 patch applied to `.github/workflows/instagram_constituency_test.yml`; validation run `28414647932`, success; artifact ID `7968986389`. |
| Member profile metrics cutover | P18 patch applied to `.github/workflows/build_member_profile_metrics_2025.yml`; first validation run `28414649704` failed in removed legacy vote rebuild step; corrected run `28414678714`, success. |
| Post-cutover monitoring | P19 document `docs/oireachtas_post_cutover_monitoring_plan.md`. |

## Applied consumer changes

### Instagram constituency renderer

The workflow now sets:

```yaml
      INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
```

Validation:

```text
Workflow ID: 261945698
Run ID: 28414647932
Run number: 5
Status: success
Artifact: instagram-constituency-test
Artifact ID: 7968986389
```

### Member profile metrics

The workflow now sets:

```yaml
      MEMBERS_INPUT_KEY: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
      MEMBER_VOTES_INPUT_KEY: "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv"
```

The legacy vote-record rebuild step was removed from this workflow because the metrics build now reads the compat vote dataset directly.

Validation:

```text
Workflow ID: 266755732
Failed run ID: 28414649704
Failed reason: legacy vote-record rebuild failed before metrics step
Corrected run ID: 28414678714
Run number: 4
Status: success
```

## Current unified compatibility keys

```text
processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv
processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv
processed/oireachtas_unified/compat/members/parquets/member_profile_metrics_2025_trial.parquet
```

## Refreshed row-count comparison

| Dataset | Legacy rows | Compat/trial rows | Current result |
|---|---:|---:|---|
| roster | 176 | 174 | 174 matched keys, 2 legacy-only keys, 0 compat-only keys. |
| member votes | 30,968 | 29,805 | 173 matched member-code keys, 0 legacy-only keys, 0 compat-only keys. |
| member profile metrics | 174 | 174 | 172 matched member codes, 2 legacy-only, 2 trial-only. |

## Remaining member-code mismatches

| Dataset | Side | Member code | Name | Party | Constituency |
|---|---|---|---|---|---|
| roster | legacy-only | `Catherine-Connolly.D.2016-10-03` | Catherine Connolly | Independent | Galway West |
| roster | legacy-only | `Paschal-Donohoe.S.2007-07-23` | Paschal Donohoe | Fine Gael | Dublin Central |
| member profile metrics | legacy-only | `Catherine-Connolly.D.2016-10-03` | Catherine Connolly | Independent | Galway West |
| member profile metrics | legacy-only | `Paschal-Donohoe.S.2007-07-23` | Paschal Donohoe | Fine Gael | Dublin Central |
| member profile metrics | trial-only | `Daniel-Ennis.D.2026-05-25` | Daniel Ennis | Social Democrats | Dublin Central |
| member profile metrics | trial-only | `Seán-Kyne.D.2011-03-09` | Seán Kyne | Fine Gael | Galway West |

Interpretation: the mismatch pattern is consistent with source freshness/member lifecycle differences rather than a deterministic build failure.

## Rollback

Rollback is workflow-config only.

Instagram rollback: remove this line from `.github/workflows/instagram_constituency_test.yml`:

```yaml
      INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
```

Member profile metrics rollback: restore legacy input keys and, if needed, restore the legacy vote-record rebuild step:

```yaml
      MEMBERS_INPUT_KEY: "raw/members/oireachtas_members_34th_dail.csv"
      MEMBER_VOTES_INPUT_KEY: "processed/votes/dail_vote_member_records.csv"
```

Detailed rollback and monitoring checks are in:

```text
docs/oireachtas_post_cutover_monitoring_plan.md
```
