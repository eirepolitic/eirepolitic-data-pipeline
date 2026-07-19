# Decisions

## Pilot grain

Use `constituency` as the first Phase 2 grain.

## Slide sequence

Use two slides:

1. generated constituency cover card;
2. classified issue ranking chart.

Both slides use the existing `title_text_media_v1` layout. The issue chart reuses `horizontal_bar_draft_v1`.

## Scenario policy

- Minimum and maximum are synthetic layout tests and cannot be published.
- The real example is internally consistent actual data selected by median complexity.
- All scenarios remain `no_publication` until explicit approval.

## Data policy

Resolve unified compatibility keys through the production pointer, retain legacy fallback keys, and record join coverage in the project manifest.

## Scope boundary

This implementation validates one constituency pilot. It does not yet provide a generic scenario builder for every project/grain and does not enable batch generation.
