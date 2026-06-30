# Oireachtas post-cutover validation summary

**Status:** validated  
**Last updated:** 2026-06-30

## P20 — generated output verification

| Consumer | Workflow ID | Run | Result | Artifact |
|---|---:|---:|---|---|
| Instagram constituency renderer | `261945698` | `28414647932` | success | `instagram-constituency-test`, artifact ID `7968986389` |
| Member profile metrics | `266755732` | `28414678714` | success | no artifact expected |

The first member-profile validation attempt, run `28414649704`, failed before metrics generation in the old legacy vote-record rebuild step. That step was removed because the workflow now reads unified compatibility vote records directly. The corrected run `28414678714` succeeded.

## P21 — post-cutover comparison reruns

| Check | Workflow ID | Run | Result |
|---|---:|---:|---|
| Compatibility adapter comparison | `294874693` | `28414819264` | success |
| Mismatch review, first rerun | `297343766` | `28414820972` | build success, review-branch publish failure |
| Mismatch review, clean rerun | `297343766` | `28414847238` | success |

## Latest comparison result

From `review/compat_adapter_comparison/latest/sample.csv`:

| Comparison | Legacy rows | Compat rows | Matched keys | Legacy-only | Compat-only |
|---|---:|---:|---:|---:|---:|
| roster | 176 | 174 | 174 | 2 | 0 |
| member votes | 30,968 | 29,805 | 173 | 0 | 0 |

## Latest mismatch result

From `review/member_code_mismatch_review/latest/manifest.json`:

| Dataset | Legacy members | Unified members | Matched | Legacy-only | Unified-only |
|---|---:|---:|---:|---:|---:|
| roster | 176 | 174 | 174 | 2 | 0 |
| member_profile_metrics | 174 | 174 | 174 | 0 | 0 |

Remaining row-level mismatches from `review/member_code_mismatch_review/latest/sample.csv`:

| Dataset | Side | Member code | Name | Party | Constituency |
|---|---|---|---|---|---|
| roster | legacy-only | `Catherine-Connolly.D.2016-10-03` | Catherine Connolly | Independent | Galway West |
| roster | legacy-only | `Paschal-Donohoe.S.2007-07-23` | Paschal Donohoe | Fine Gael | Dublin Central |

## Interpretation

The member-profile metrics output now matches the unified trial set exactly by member code. The remaining mismatch is limited to the roster source difference already known from prior validation.

## Current status

The applied pre-production cutovers are validated. Continue monitoring with the rollback plan in `docs/oireachtas_post_cutover_monitoring_plan.md`.
