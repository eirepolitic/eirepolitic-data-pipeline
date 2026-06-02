# `_discovery` sample

| endpoint_name | endpoint | ok | status_code | elapsed_seconds | result_count | top_keys | result_wrapper_keys | schema_hash | error | url |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| houses | /houses | True | 200 | 0.79 | 5 | head,results | house | a586cf54890aab8f | None | https://api.oireachtas.ie/v1/houses?limit=5 |
| members | /members | True | 200 | 0.408 | 5 | head,results | member | 732c6689d2769190 | None | https://api.oireachtas.ie/v1/members?chamber=dail&house_no=34&limit=5 |
| debates | /debates | True | 200 | 0.832 | 5 | head,results | contextDate,debateRecord | fa8ed8721b80eca9 | None | https://api.oireachtas.ie/v1/debates?chamber_id=%2Fie%2Foireachtas%2Fhouse%2Fdail%2F34&lang=en&limit=5 |
| divisions | /divisions | True | 200 | 0.11 | 3 | head,results | contextDate,division | 99138f2da33a4956 | None | https://api.oireachtas.ie/v1/divisions?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=5 |
| votes_fallback_probe | /votes | True | 200 | 0.393 | 3 | head,results | contextDate,division | 99138f2da33a4956 | None | https://api.oireachtas.ie/v1/votes?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=5 |
| questions | /questions | True | 200 | 0.247 | 5 | head,results | contextDate,question | 156f608009af96a9 | None | https://api.oireachtas.ie/v1/questions?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=5 |
| legislation | /legislation | True | 200 | 0.111 | 5 | head,results | bill,billSort,contextDate | 1051ad4abc57aa70 | None | https://api.oireachtas.ie/v1/legislation?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=5 |
| parties | /parties | True | 200 | 0.108 | 11 | head,results | house,party | f5544763061f1568 | None | https://api.oireachtas.ie/v1/parties?limit=5 |
| constituencies | /constituencies | True | 200 | 0.246 | 8 | head,results | constituencyOrPanel,house | 2d4358b21d593e39 | None | https://api.oireachtas.ie/v1/constituencies?limit=5 |
