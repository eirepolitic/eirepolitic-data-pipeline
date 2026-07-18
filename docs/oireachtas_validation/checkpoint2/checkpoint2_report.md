# Oireachtas validation — Checkpoint 2

## Scorecard

| Table | Live rows | Fresh official rows | Tests | Passed | Failed |
|---|---:|---:|---:|---:|---:|
| silver_houses | 68 | 68 | 16 | 16 | 0 |
| silver_constituencies | 43 | 43 | 16 | 16 | 0 |
| silver_parties | 11 | 11 | 16 | 16 | 0 |
| silver_members | 176 | 176 | 17 | 17 | 0 |
| silver_member_memberships | 176 | 176 | 17 | 12 | 5 |
| silver_member_parties | 280 | 178 | 17 | 11 | 6 |
| silver_member_constituencies | 276 | 176 | 17 | 11 | 6 |
| silver_member_offices | 123 | 77 | 17 | 11 | 6 |
| cross_table | 0 | 0 | 8 | 8 | 0 |

## Findings

| Table | Test | Expected | Actual | Sample | Details |
|---|---|---|---|---|---|
| silver_member_parties | at_most_one_current_row_per_member | 0 | 98 |  | {"Aidan-Farrelly.D.2024-11-29":2,"Aisling-Dempsey.D.2024-11-29":2,"Alan-Dillon.D.2020-02-08":2,"Alan-Kelly.S.2007-07-23":2,"Albert-Dolan.D.2024-11-29":2,"Ann-Graves.D.2024-11-29":2,"Barry-Heneghan.D.2024-11-29":2,"Brian-Brennan.D.2024-11-29":2,"Cathal-Crowe.D.2020-02-08":2,"Catherine-Ardagh.S.2016-04-25":2} |
| silver_member_constituencies | at_most_one_current_row_per_member | 0 | 98 |  | {"Aidan-Farrelly.D.2024-11-29":2,"Aisling-Dempsey.D.2024-11-29":2,"Alan-Dillon.D.2020-02-08":2,"Alan-Kelly.S.2007-07-23":2,"Albert-Dolan.D.2024-11-29":2,"Ann-Graves.D.2024-11-29":2,"Barry-Heneghan.D.2024-11-29":2,"Brian-Brennan.D.2024-11-29":2,"Cathal-Crowe.D.2020-02-08":2,"Catherine-Ardagh.S.2016-04-25":2} |
| silver_member_offices | at_most_one_current_row_per_member | 0 | 26 |  | {"Alan-Dillon.D.2020-02-08":2,"Catherine-Ardagh.S.2016-04-25":2,"Colm-Brophy.D.2016-10-03":2,"Dara-Calleary.D.2007-06-14":4,"Darragh-O'Brien.D.2007-06-14":2,"Emer-Higgins.D.2020-02-08":2,"Frank-Feighan.S.2002-09-12":2,"Helen-McEntee.D.2013-03-27":2,"Jack-Chambers.D.2016-10-03":2,"James-Browne.D.2016-10-03":2} |
| silver_member_memberships | official_api_sample_match | all selected stable fields equal | mismatch | https://data.oireachtas.ie/ie/oireachtas/member/id/Brendan-Smith.D.1992-12-14/house/dail/34 | {"membership_end": {"live": "", "official": "nan"}} |
| silver_member_memberships | official_api_sample_match | all selected stable fields equal | mismatch | https://data.oireachtas.ie/ie/oireachtas/member/id/Christopher-O'Sullivan.D.2020-02-08/house/dail/34 | {"membership_end": {"live": "", "official": "nan"}} |
| silver_member_memberships | official_api_sample_match | all selected stable fields equal | mismatch | https://data.oireachtas.ie/ie/oireachtas/member/id/Thomas-Byrne.D.2007-06-14/house/dail/34 | {"membership_end": {"live": "", "official": "nan"}} |
| silver_member_memberships | official_api_sample_match | all selected stable fields equal | mismatch | https://data.oireachtas.ie/ie/oireachtas/member/id/Maeve-O'Connell.D.2024-11-29/house/dail/34 | {"membership_end": {"live": "", "official": "nan"}} |
| silver_member_memberships | official_api_sample_match | all selected stable fields equal | mismatch | https://data.oireachtas.ie/ie/oireachtas/member/id/Robert-Troy.D.2011-03-09/house/dail/34 | {"membership_end": {"live": "", "official": "nan"}} |
| silver_member_parties | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_party:cf76c01f1bd520be | {"party_end": {"live": "", "official": "nan"}} |
| silver_member_parties | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_party:337fb14911713f89 | {"party_end": {"live": "", "official": "nan"}} |
| silver_member_parties | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_party:490ddb3b9259375c | {"party_end": {"live": "", "official": "nan"}} |
| silver_member_parties | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_party:d7441dba5fd70487 | {"party_end": {"live": "", "official": "nan"}} |
| silver_member_parties | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_party:ce20e03578118f22 | {"party_end": {"live": "", "official": "nan"}} |
| silver_member_constituencies | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_constituency:7f308817fd7b4116 | {"represent_end": {"live": "", "official": "nan"}} |
| silver_member_constituencies | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_constituency:d179706178052c45 | {"represent_end": {"live": "", "official": "nan"}} |
| silver_member_constituencies | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_constituency:85006a837d95990e | {"represent_end": {"live": "", "official": "nan"}} |
| silver_member_constituencies | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_constituency:146e2114fa04f088 | {"represent_end": {"live": "", "official": "nan"}} |
| silver_member_constituencies | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_constituency:2998363e9d8c0728 | {"represent_end": {"live": "", "official": "nan"}} |
| silver_member_offices | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_office:bab9f7efe5aa8c7f | {"office_end": {"live": "", "official": "nan"}} |
| silver_member_offices | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_office:d3f41b6ffa45c513 | {"office_end": {"live": "", "official": "nan"}} |
| silver_member_offices | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_office:1d4b76b663e7567a | {"office_end": {"live": "", "official": "nan"}} |
| silver_member_offices | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_office:77b2715436c127eb | {"office_end": {"live": "", "official": "nan"}} |
| silver_member_offices | official_api_sample_match | all selected stable fields equal | mismatch | generated:member_office:0027feca877a8dab | {"office_end": {"live": "", "official": "nan"}} |

## Official API sample checks

- `silver_houses` / `https://data.oireachtas.ie/ie/oireachtas/house/seanad/2`: **pass** — https://api.oireachtas.ie/v1/houses?limit=200&skip=0
- `silver_houses` / `https://data.oireachtas.ie/ie/oireachtas/house/dail/26`: **pass** — https://api.oireachtas.ie/v1/houses?limit=200&skip=0
- `silver_houses` / `https://data.oireachtas.ie/ie/oireachtas/house/seanad/1928`: **pass** — https://api.oireachtas.ie/v1/houses?limit=200&skip=0
- `silver_houses` / `https://data.oireachtas.ie/ie/oireachtas/house/dail/7`: **pass** — https://api.oireachtas.ie/v1/houses?limit=200&skip=0
- `silver_houses` / `https://data.oireachtas.ie/ie/oireachtas/house/dail/18`: **pass** — https://api.oireachtas.ie/v1/houses?limit=200&skip=0
- `silver_constituencies` / `https://data.oireachtas.ie/ie/oireachtas/house/dail/34/constituency/Dublin-South-West`: **pass** — https://api.oireachtas.ie/v1/constituencies?limit=200&chamber=dail&house_no=34&skip=0
- `silver_constituencies` / `https://data.oireachtas.ie/ie/oireachtas/house/dail/34/constituency/Wicklow-Wexford`: **pass** — https://api.oireachtas.ie/v1/constituencies?limit=200&chamber=dail&house_no=34&skip=0
- `silver_constituencies` / `https://data.oireachtas.ie/ie/oireachtas/house/dail/34/constituency/Mayo`: **pass** — https://api.oireachtas.ie/v1/constituencies?limit=200&chamber=dail&house_no=34&skip=0
- `silver_constituencies` / `https://data.oireachtas.ie/ie/oireachtas/house/dail/34/constituency/Dún-Laoghaire`: **pass** — https://api.oireachtas.ie/v1/constituencies?limit=200&chamber=dail&house_no=34&skip=0
- `silver_constituencies` / `https://data.oireachtas.ie/ie/oireachtas/house/dail/34/constituency/Louth`: **pass** — https://api.oireachtas.ie/v1/constituencies?limit=200&chamber=dail&house_no=34&skip=0
- `silver_parties` / `https://data.oireachtas.ie/ie/oireachtas/party/dail/34/Independent`: **pass** — https://api.oireachtas.ie/v1/parties?limit=200&chamber=dail&house_no=34&skip=0
- `silver_parties` / `https://data.oireachtas.ie/ie/oireachtas/party/dail/34/People_Before_Profit_Solidarity`: **pass** — https://api.oireachtas.ie/v1/parties?limit=200&chamber=dail&house_no=34&skip=0
- `silver_parties` / `https://data.oireachtas.ie/ie/oireachtas/party/dail/34/Social_Democrats`: **pass** — https://api.oireachtas.ie/v1/parties?limit=200&chamber=dail&house_no=34&skip=0
- `silver_parties` / `https://data.oireachtas.ie/ie/oireachtas/party/dail/34/Aontú`: **pass** — https://api.oireachtas.ie/v1/parties?limit=200&chamber=dail&house_no=34&skip=0
- `silver_parties` / `https://data.oireachtas.ie/ie/oireachtas/party/dail/34/Labour_Party`: **pass** — https://api.oireachtas.ie/v1/parties?limit=200&chamber=dail&house_no=34&skip=0
- `silver_members` / `Robert-Troy.D.2011-03-09`: **pass** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_members` / `Pádraig-O'Sullivan.D.2019-11-29`: **pass** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_members` / `Liam-Quaide.D.2024-11-29`: **pass** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_members` / `Darragh-O'Brien.D.2007-06-14`: **pass** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_members` / `Conor-D-McGuinness.D.2024-11-29`: **pass** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_memberships` / `https://data.oireachtas.ie/ie/oireachtas/member/id/Brendan-Smith.D.1992-12-14/house/dail/34`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_memberships` / `https://data.oireachtas.ie/ie/oireachtas/member/id/Christopher-O'Sullivan.D.2020-02-08/house/dail/34`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_memberships` / `https://data.oireachtas.ie/ie/oireachtas/member/id/Thomas-Byrne.D.2007-06-14/house/dail/34`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_memberships` / `https://data.oireachtas.ie/ie/oireachtas/member/id/Maeve-O'Connell.D.2024-11-29/house/dail/34`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_memberships` / `https://data.oireachtas.ie/ie/oireachtas/member/id/Robert-Troy.D.2011-03-09/house/dail/34`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_parties` / `generated:member_party:cf76c01f1bd520be`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_parties` / `generated:member_party:337fb14911713f89`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_parties` / `generated:member_party:490ddb3b9259375c`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_parties` / `generated:member_party:d7441dba5fd70487`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_parties` / `generated:member_party:ce20e03578118f22`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_constituencies` / `generated:member_constituency:7f308817fd7b4116`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_constituencies` / `generated:member_constituency:d179706178052c45`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_constituencies` / `generated:member_constituency:85006a837d95990e`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_constituencies` / `generated:member_constituency:146e2114fa04f088`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_constituencies` / `generated:member_constituency:2998363e9d8c0728`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_offices` / `generated:member_office:bab9f7efe5aa8c7f`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_offices` / `generated:member_office:d3f41b6ffa45c513`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_offices` / `generated:member_office:1d4b76b663e7567a`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_offices` / `generated:member_office:77b2715436c127eb`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
- `silver_member_offices` / `generated:member_office:0027feca877a8dab`: **fail** — https://api.oireachtas.ie/v1/members?limit=200&chamber=dail&house_no=34&skip=0
