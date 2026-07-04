# Oireachtas final post-cutover validation sweep

**Status:** complete  
**Last updated:** 2026-07-04

## Purpose

Rerun downstream validation after all controlled pre-production cutovers for deterministic compatibility outputs and enrichment compatibility outputs.

## Validation runs

### Compatibility adapter comparison

```text
Workflow ID: 294874693
Run ID: 28691936308
Run number: 5
Result: success
Artifact ID: 8077413255
```

Latest review sample:

```text
members_roster_compat: pass
member_votes_compat: pass
```

Known roster difference remains unchanged:

```text
legacy rows: 176
compat rows: 174
matched member codes: 174
legacy-only member codes: 2
compat-only member codes: 0
```

Vote compat remains structurally valid:

```text
legacy rows: 30968
compat rows: 29805
matched key count: 173
legacy-only key count: 0
compat-only key count: 0
```

### Member-code mismatch review

```text
Workflow ID: 297343766
Run ID: 28691938402
Run number: 5
Result: success
Artifact ID: 8077412883
```

Latest mismatch sample remains limited to the two known roster-only legacy members:

```text
Catherine Connolly — Independent — Galway West
Paschal Donohoe — Fine Gael — Dublin Central
```

No member-profile metrics member-code mismatches are indicated by the latest member-profile trial evidence.

## Conclusion

Final post-cutover validation passed. The only remaining known member-code caveat is the pre-existing two-member roster delta between legacy roster and unified/compat roster.
