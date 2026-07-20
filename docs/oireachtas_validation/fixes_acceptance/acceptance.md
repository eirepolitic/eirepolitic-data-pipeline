# Oireachtas validation-fixes candidate acceptance

- Batch: `validation-fixes-20260719-1`
- Overall: **fail**

| Check | Status | Details |
|---|---|---|
| batch_manifest_validated | **pass** | {"status": "validated", "table_count": 32, "validation": {"duplicate_tables": [], "failed_tables": [], "missing_objects": [], "missing_tables": []}} |
| member_party_business_keys_unique | **pass** | {"duplicate_rows": 0} |
| member_constituency_business_keys_unique | **pass** | {"duplicate_rows": 0} |
| current_party_values_unchanged | **pass** | {"difference_count": 0, "samples": []} |
| current_constituency_values_unchanged | **pass** | {"difference_count": 0, "samples": []} |
| recent_official_debate_sections_present | **pass** | {"candidate_rows": 5008, "missing_count": 0, "missing_samples": [], "official_rows": 106, "source": "https://api.oireachtas.ie/v1/debates?chamber_id=%2Fie%2Foireachtas%2Fhouse%2Fdail%2F34&lang=en&date_start=2026-07-14&date_end=2026-07-19&limit=200&skip=0"} |
| recent_official_questions_present | **pass** | {"candidate_rows": 117042, "missing_count": 0, "missing_samples": [], "official_rows": 2032, "source": "https://api.oireachtas.ie/v1/questions?chamber=dail&house_no=34&date_start=2026-07-14&date_end=2026-07-19&limit=200&skip=0"} |
| official_bill_versions_present | **pass** | {"candidate_rows": 508, "missing_count": 0, "missing_samples": [], "official_rows": 508, "source": "https://api.oireachtas.ie/v1/legislation?chamber=dail&house_no=34&date_start=2024-11-29&date_end=2026-07-19&limit=200&skip=0"} |
| official_bill_debate_business_rows_present | **pass** | {"candidate_rows": 1222, "missing_business_rows": 0, "missing_samples": [], "official_rows_for_candidate_bills": 1172, "source": "https://api.oireachtas.ie/v1/legislation?chamber=dail&house_no=34&date_start=2024-11-29&date_end=2026-07-19&limit=200&skip=0"} |
| control_manifest_counts_and_schemas_match_candidate | **fail** | {"failure_count": 4, "failure_samples": [{"differences": {"row_count": {"actual_csv": 1222, "actual_parquet": 1222, "stored": 1172}}, "table": "silver_bill_debates"}, {"differences": {"row_count": {"actual_csv": 123, "actual_parquet": 123, "stored": 77}}, "table": "silver_member_offices"}, {"differences": {"row_count": {"actual_csv": 3480, "actual_parquet": 3480, "stored": 3455}}, "table": "silver_source_files"}, {"differences": {"row_count": {"actual_csv": 66147, "actual_parquet": 66147, "stored": 66137}}, "table": "silver_speeches"}], "missing_tables": [], "row_count": 31} |
