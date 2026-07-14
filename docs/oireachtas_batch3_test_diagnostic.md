# Oireachtas Batch 3 test diagnostic

- Run ID: 29311589162
- Commit: eaca6c2c24124efe941b54f03d054bd2369c73eb
- Exit code: 1

```text
test_discovery_can_request_one_page_only (test_oireachtas_pagination.OireachtasPaginationTests.test_discovery_can_request_one_page_only) ... ok
test_merges_pages_until_reported_total (test_oireachtas_pagination.OireachtasPaginationTests.test_merges_pages_until_reported_total) ... ok
test_repeated_page_fails_instead_of_looping (test_oireachtas_pagination.OireachtasPaginationTests.test_repeated_page_fails_instead_of_looping) ... ok
test_short_page_completes_when_total_is_unavailable (test_oireachtas_pagination.OireachtasPaginationTests.test_short_page_completes_when_total_is_unavailable) ... ok
test_both_switches_are_required (test_oireachtas_repair_regressions.ProductionPublishingGuardTests.test_both_switches_are_required) ... ok
test_guard_suppresses_mutable_latest_and_compat_writes (test_oireachtas_repair_regressions.ProductionPublishingGuardTests.test_guard_suppresses_mutable_latest_and_compat_writes) ... ok
test_production_publish_is_default_deny (test_oireachtas_repair_regressions.ProductionPublishingGuardTests.test_production_publish_is_default_deny) ... ok
test_compatibility_dq_fails_when_legacy_keys_are_missing (test_oireachtas_repair_regressions.RemainingConfirmedRegressionTests.test_compatibility_dq_fails_when_legacy_keys_are_missing) ... expected failure
test_overlapping_incremental_windows_preserve_and_update_history (test_oireachtas_write_semantics.MergeSemanticsTests.test_overlapping_incremental_windows_preserve_and_update_history) ... ok
test_snapshot_replace_does_not_retain_missing_rows (test_oireachtas_write_semantics.MergeSemanticsTests.test_snapshot_replace_does_not_retain_missing_rows) ... ok
test_yearly_aggregation_uses_preserved_history (test_oireachtas_write_semantics.MergeSemanticsTests.test_yearly_aggregation_uses_preserved_history) ... ok
test_constituency_id_ignores_later_end_date (test_oireachtas_write_semantics.StableHistoryIdentityTests.test_constituency_id_ignores_later_end_date) ... ok
test_membership_id_ignores_later_end_date (test_oireachtas_write_semantics.StableHistoryIdentityTests.test_membership_id_ignores_later_end_date) ... ok
test_office_id_ignores_later_end_date (test_oireachtas_write_semantics.StableHistoryIdentityTests.test_office_id_ignores_later_end_date) ... ok
test_party_id_ignores_later_end_date (test_oireachtas_write_semantics.StableHistoryIdentityTests.test_party_id_ignores_later_end_date) ... ok
test_foreign_key_orphans_are_visible (test_oireachtas_write_semantics.TemporalAndIntegrityTests.test_foreign_key_orphans_are_visible) ... ok
test_future_open_ended_record_is_not_current (test_oireachtas_write_semantics.TemporalAndIntegrityTests.test_future_open_ended_record_is_not_current) ... ok
test_overlapping_ranges_are_counted (test_oireachtas_write_semantics.TemporalAndIntegrityTests.test_overlapping_ranges_are_counted) ... ok
test_temporal_integrity_rejects_invalid_and_future_current_rows (test_oireachtas_write_semantics.TemporalAndIntegrityTests.test_temporal_integrity_rejects_invalid_and_future_current_rows) ... ERROR
test_every_registered_table_has_write_policy (test_oireachtas_write_semantics.WritePolicyCoverageTests.test_every_registered_table_has_write_policy) ... ok

======================================================================
ERROR: test_temporal_integrity_rejects_invalid_and_future_current_rows (test_oireachtas_write_semantics.TemporalAndIntegrityTests.test_temporal_integrity_rejects_invalid_and_future_current_rows)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/runner/work/eirepolitic-data-pipeline/eirepolitic-data-pipeline/tests/test_oireachtas_write_semantics.py", line 72, in test_temporal_integrity_rejects_invalid_and_future_current_rows
    result = temporal_integrity(frame, policy=policy, as_of=date(2026, 7, 13))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/eirepolitic-data-pipeline/eirepolitic-data-pipeline/extract/oireachtas/merge.py", line 43, in temporal_integrity
    invalid_ranges = sum(
                     ^^^^
  File "/home/runner/work/eirepolitic-data-pipeline/eirepolitic-data-pipeline/extract/oireachtas/merge.py", line 46, in <genexpr>
    if start is not None and end is not None and start > end
                                                 ^^^^^^^^^^^
TypeError: '>' not supported between instances of 'str' and 'float'

----------------------------------------------------------------------
Ran 20 tests in 0.047s

FAILED (errors=1, expected failures=1)
```
