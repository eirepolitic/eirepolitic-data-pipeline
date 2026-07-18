# Oireachtas validation — Checkpoint 2 conclusion

## Overall result

Checkpoint 2 passed for houses, constituencies, parties, members, memberships, relationships, and official-API identity checks, with one confirmed data-quality defect in two historical bridge tables.

## Confirmed passes

- `silver_houses`: schema, keys, dates, CSV/Parquet parity, official API samples.
- `silver_constituencies`: schema, keys, dates, CSV/Parquet parity, official API samples.
- `silver_parties`: schema, keys, dates, CSV/Parquet parity, official API samples.
- `silver_members`: schema, keys, dates, current-member reconciliation, official API samples.
- `silver_member_memberships`: schema, keys, dates, member/house relationships, official API samples.
- All configured member/reference foreign-key checks passed.
- No conflicting current party or constituency values were found.

## Confirmed defect

### `silver_member_parties`

- 280 total rows.
- 272 rows marked current.
- 98 members have repeated current rows.
- All 98 cases repeat the same party value.
- 196 rows are exact duplicates on the business fields: member, party, start date and end date.
- No member has conflicting current party values.

### `silver_member_constituencies`

- 276 total rows.
- 272 rows marked current.
- 98 members have repeated current rows.
- All 98 cases repeat the same constituency value.
- 196 rows are exact duplicates on the business fields: member, constituency, start date and end date.
- No member has conflicting current constituency values.

This is classified as a pipeline merge/identity defect rather than a disagreement with the official source. The two table builders create different row IDs for records that are otherwise the same business relationship, allowing both copies to survive an upsert.

## Valid multiple rows

### `silver_member_offices`

- 123 total rows.
- 68 rows marked current.
- 26 members have multiple current office rows.
- Every case contains distinct office values.
- No exact business duplicates were found.

These are classified as valid simultaneous roles, not defects.

## Validator note

External sample mismatches where one side contained an empty string and the other contained pandas `NaN` are comparison-normalization false positives. Both represent a missing end date and do not indicate a source mismatch.

## Disposition

- Production was not modified.
- The duplicate party and constituency rows remain listed as a repair item.
- Validation can continue because the duplicates do not create conflicting current values and all relationships still resolve.
