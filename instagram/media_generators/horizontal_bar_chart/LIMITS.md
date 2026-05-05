# Horizontal Bar Chart Limits

## Recommended

- 5–10 categories.
- Label length under 32 characters.
- Width at least 800 px and height at least 600 px.
- Use dark or light EirePolitic palettes only until more palettes are tested.

## Stress-tested in fake cases

- 20 categories render, but visual density is high.
- Labels above 42 characters are flagged in `warnings`.
- Zero and decimal values render.
- Non-numeric values are coerced to `0` and warned.

## Known failure modes

- Long labels can overlap or be clipped in very small widths.
- Negative values render but are flagged because most issue/count visuals should not use them.
- More than 12 categories should usually become a ranking/table card instead.
