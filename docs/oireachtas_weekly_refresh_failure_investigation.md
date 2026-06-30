# Oireachtas weekly refresh failure investigation

**Status:** root cause patched; safe validation run in progress  
**Last updated:** 2026-06-30

## Failed runs investigated

| Workflow | Run | Event | Result |
|---|---:|---|---|
| Oireachtas Weekly Refresh | `27898282130` | schedule | failure |
| Oireachtas Weekly Refresh | `28314747505` | schedule | failure |

Latest failed scheduled run inspected:

```text
28314747505
```

## Failure location

The job failed in this step:

```text
Run weekly table set
```

The setup, checkout, dependency install, review branch publish, summary, and artifact upload steps all completed successfully.

## Root cause

The weekly run completed these member tables successfully before failing:

```text
silver_members
silver_member_memberships
silver_member_parties
silver_member_constituencies
silver_member_offices
```

The next table was:

```text
silver_debate_records
```

The `silver_debate_records` DQ failed because recent debate records had XML links but not PDF links.

Failing DQ checks from the review output:

```text
source_pdf_uri_populated: fail
source_file_ids_populated: fail
```

The output still had valid debate rows and populated XML links:

```text
row_count_gt_zero: pass
source_xml_uri_populated: pass
```

## Fix applied

File patched:

```text
extract/oireachtas/table_debate_records.py
```

Change:

- XML source links remain required.
- PDF source links are now optional.
- PDF presence is recorded as informational metrics:

```text
pdf_present_count
pdf_missing_count
```

New DQ checks:

```text
source_xml_uri_populated
source_pdf_uri_optional
source_file_id_xml_populated
source_file_id_pdf_consistent_when_present
```

## Why this is correct

A missing PDF link should not fail the debate record table if the debate record itself is valid and the XML source link is present. Some recent Oireachtas debate records expose XML without PDF.

## Validation

A safe manual weekly validation run was dispatched after the patch:

```text
Workflow ID: 294426406
Run ID: 28421557467
Mode: workflow_dispatch defaults
Expected behavior: test mode, low limits, latest publishing suppressed by test mode
```

Current status at documentation time:

```text
in_progress
```

The run progressed past the original quick failure point and remained active in the table-set step, indicating the immediate PDF-DQ blocker was cleared. Final run status should be checked before relying on the next scheduled weekly run.
