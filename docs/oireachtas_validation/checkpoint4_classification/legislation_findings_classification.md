# Legislation findings classification

## Comparable official scope

| Table | Live | Official for live bills | Shared | Missing from live | Extra in live |
|---|---:|---:|---:|---:|---:|
| silver_bills | 404 | 404 | 404 | 0 | 0 |
| silver_bill_versions | 506 | 507 | 506 | 1 | 0 |
| silver_bill_stages | 1358 | 1358 | 1358 | 0 | 0 |
| silver_bill_related_docs | 345 | 345 | 345 | 0 | 0 |
| silver_bill_sponsors | 1211 | 1211 | 1211 | 0 | 0 |
| silver_bill_debates | 1215 | 1171 | 1165 | 6 | 50 |
| silver_bill_events | 692 | 692 | 692 | 0 | 0 |

## Sponsor classification

Bills with multiple primary sponsors: 2

- `https://data.oireachtas.ie/ie/oireachtas/bill/2016/31`: live=[{"sponsor_name": "Joan Collins", "sponsor_uri": "https://data.oireachtas.ie/ie/oireachtas/member/id/Joan-Collins.D.2011-03-09", "sponsor_role_name": "", "sponsor_order": "15"}, {"sponsor_name": "Gerry Adams", "sponsor_uri": "https://data.oireachtas.ie/ie/oireachtas/member/id/Gerry-Adams.D.2011-03-09", "sponsor_role_name": "", "sponsor_order": "27"}]; official=[{"sponsor_name": "Joan Collins", "sponsor_uri": "https://data.oireachtas.ie/ie/oireachtas/member/id/Joan-Collins.D.2011-03-09", "sponsor_role_name": NaN, "sponsor_order": "15"}, {"sponsor_name": "Gerry Adams", "sponsor_uri": "https://data.oireachtas.ie/ie/oireachtas/member/id/Gerry-Adams.D.2011-03-09", "sponsor_role_name": NaN, "sponsor_order": "27"}]
- `https://data.oireachtas.ie/ie/oireachtas/bill/2025/4`: live=[{"sponsor_name": "Paul Murphy", "sponsor_uri": "https://data.oireachtas.ie/ie/oireachtas/member/id/Paul-Murphy.D.2014-10-10", "sponsor_role_name": "", "sponsor_order": "1"}, {"sponsor_name": "Paul Murphy", "sponsor_uri": "https://data.oireachtas.ie/ie/oireachtas/member/id/Paul-Murphy.D.2014-10-10", "sponsor_role_name": "", "sponsor_order": "4"}]; official=[{"sponsor_name": "Paul Murphy", "sponsor_uri": "https://data.oireachtas.ie/ie/oireachtas/member/id/Paul-Murphy.D.2014-10-10", "sponsor_role_name": NaN, "sponsor_order": "1"}, {"sponsor_name": "Paul Murphy", "sponsor_uri": "https://data.oireachtas.ie/ie/oireachtas/member/id/Paul-Murphy.D.2014-10-10", "sponsor_role_name": NaN, "sponsor_order": "4"}]

## Debate-link classification

- Comparable current-Dáil links: 419
- Missing comparable debate records: 0
- Missing comparable sections using `(debate_id, section_eid)`: 0
- Out-of-scope Seanad/older links: 796

## Historical bill scope

- Bills introduced before 2024-11-29: 239
- Older bills active/updated after start: 239
- Older bills whose last event also predates start: 0

## Classifications

- House URI failures are validator-model errors: legislation uses chamber-definition URIs, while `silver_houses` uses numbered house-instance URIs.
- Bill section links must join on `(debate_id, section_eid)`.
- Multiple primary sponsors are permitted when present in the official payload.
- Raw official row-count differences are not comparable until restricted to the same live bill IDs.
