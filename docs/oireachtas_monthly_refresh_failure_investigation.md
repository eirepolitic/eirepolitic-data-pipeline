# Oireachtas monthly scheduled refresh failure investigation

**Status:** complete  
**Last updated:** 2026-07-05

## Failed scheduled run

```text
Workflow: Oireachtas Monthly Refresh
Workflow ID: 294432002
Run ID: 28504651002
Event: schedule
Result: failure
Created: 2026-07-01T08:36:19Z
Failed step: Run monthly table set
Artifact ID: 8004493556
```

## Investigation method

The uploaded artifact metadata was available, but detailed artifact contents were not directly readable through the available repository action API. The investigation used review output published by the failed run on the `oireachtas-review-output` branch.

## Root cause sequence

The failed scheduled run successfully completed these tables before stopping:

```text
silver_constituencies
silver_parties
silver_source_files
silver_bills
silver_bill_versions
```

The first failing table was:

```text
silver_bill_stages
```

### First failure: silver_bill_stages

Failed DQ checks:

```text
house_uri_populated
house_name_populated
```

Evidence from the failed run:

```text
row_count: 189
primary_key_unique: true
house_name_values: 27th Seanad, 33rd Dáil, 34th Dáil
stage_name_values included: Cream List
```

Root cause:

```text
Live Oireachtas legislation data can include bill stage records without populated house metadata. The table itself still has stable bill/stage IDs, stage names, dates, and order values. The DQ rule was too strict for optional house metadata.
```

### Second failure discovered during validation: silver_bill_sponsors

After fixing `silver_bill_stages`, the production-like monthly validation progressed to:

```text
silver_bill_sponsors
```

Failed DQ checks:

```text
sponsor_name_populated
sponsor_uri_populated
```

Root cause:

```text
Some bill sponsor records are role-only records where sponsor.by is not populated, while sponsor.as may still describe the role. The primary key remains deterministic and the bill/sponsor order is preserved. The DQ rule was too strict for optional sponsor identity metadata.
```

### Third failure discovered during validation: silver_bill_events

After fixing `silver_bill_sponsors`, the production-like monthly validation progressed to:

```text
silver_bill_events
```

Failed DQ check:

```text
chamber_populated
```

Root cause:

```text
Some bill event records lack chamber metadata. The event URI, event type, event name, event date, bill ID, and event order remain populated. The DQ rule was too strict for optional chamber metadata.
```

## Files patched

```text
extract/oireachtas/table_bill_stages.py
extract/oireachtas/table_bill_sponsors.py
extract/oireachtas/table_bill_events.py
```

## Fix pattern

The strict DQ checks were changed to informational optional checks:

```text
silver_bill_stages: house_uri_optional, house_name_optional
silver_bill_sponsors: sponsor_name_optional, sponsor_uri_optional
silver_bill_events: chamber_optional
```

The following checks remain required:

```text
row_count_gt_zero
required_columns_present
primary_key_non_null
primary_key_unique
bill_id_populated
date/order/name fields relevant to each table
```

## Validation

A production-like manual monthly validation was run against the failed scheduled window after the patches.

```text
Workflow ID: 294432002
Run ID: 28726922946
Run number: 5
Result: success
Artifact ID: 8087552815
```

Validation window:

```text
mode: incremental
publish_latest: auto
date_start: 2026-05-25
date_end: 2026-06-30
limit: 250
```

Latest patched DQ evidence:

```text
silver_bill_stages: pass; row_count 214; house metadata missing count 3
silver_bill_sponsors: pass; row_count 60; sponsor identity missing count 30
silver_bill_events: pass; row_count 48; chamber metadata missing count 3
control_data_quality_results: pass
```

## Temporary validation change reverted

The monthly workflow manual defaults were temporarily changed to the failed scheduled window so the validation would exercise the same data. After validation, the safe manual defaults were restored.

## Conclusion

Monthly scheduled refresh failed because live legislation data exposed optional metadata that DQ treated as mandatory. The monthly refresh now passes a production-like validation run after making those fields informational.
