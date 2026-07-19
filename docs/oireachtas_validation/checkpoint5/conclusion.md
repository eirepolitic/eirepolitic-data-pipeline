# Oireachtas validation — Checkpoint 5 conclusion

## Result

Checkpoint 5 passed.

- `gold_current_members`: exact rebuild match, 174 rows.
- `gold_member_activity_yearly`: exact rebuild match, 531 rows.
- `gold_member_activity_monthly`: exact rebuild match, 3,363 rows.
- `gold_constituency_activity_yearly`: exact rebuild match, 129 rows.
- `gold_content_fact_pool`: 8,046 rows and keys matched; all source metrics resolved correctly.
- Monthly speech and vote totals reconcile exactly to annual totals.

## Formatting warning

The fact-pool row comparison found values represented as `0.0` in the live CSV and `0` in the independent rebuild. Numeric comparison against the referenced source tables passed for every fact. This is a serialization-format difference, not a metric disagreement.

## Disposition

No gold-table repair is required for correctness. Standardizing numeric string formatting in `gold_content_fact_pool.metric_value` is an optional cleanup item.
