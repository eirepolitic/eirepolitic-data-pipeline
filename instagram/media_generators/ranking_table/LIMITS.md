# Ranking Table Generator Limits

## Recommended

- 5–10 rows for 920x720 media.
- Keep names under 36 characters where possible.
- Keep sublabels under 34 characters where possible.
- Use `row_limit` to control readability before rendering.

## Stress-tested

- 20 input rows render by truncating to `row_limit`.
- Long names and long party/constituency labels are shortened with ellipses.
- Missing names fall back to `Missing name`.
- Non-numeric values are converted to 0 and reported in warnings.
- Light and dark palettes are supported.

## Known failure modes

- More than 10 rows at 920x720 becomes hard to read.
- Very long names can lose important distinguishing text when shortened.
- Smaller canvases should use `row_limit: 8` or lower.
- Multi-column ranking tables are not supported yet.

## Next improvements

- Optional avatar column.
- Configurable columns beyond rank/name/sublabel/value.
- Better tie handling.
- CSV/S3 input mode in the common generator runner.
