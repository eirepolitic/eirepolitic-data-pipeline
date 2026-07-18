# Oireachtas validation — Checkpoint 3

| Table | Live rows | Fresh official rows | Tests | Passed | Failed |
|---|---:|---:|---:|---:|---:|
| silver_source_files | 3477 | 0 | 23 | 23 | 0 |
| silver_debate_records | 164 | 164 | 16 | 16 | 0 |
| silver_debate_sections | 5001 | 5008 | 17 | 17 | 0 |
| silver_speeches | 66083 | 0 | 16 | 16 | 0 |
| silver_divisions | 398 | 398 | 15 | 15 | 0 |
| silver_division_tallies | 1194 | 1194 | 13 | 9 | 4 |
| silver_member_votes | 58868 | 58868 | 16 | 16 | 0 |
| silver_questions | 115830 | 117042 | 15 | 15 | 0 |
| cross_table | 0 | 0 | 9 | 8 | 1 |

## Findings

| Table | Test | Expected | Actual | Sample | Details |
|---|---|---|---|---|---|
| cross_table | speech_member | 0 | 1 |  | [{"speech_id":"speech:4ee771a33a97bc5f938f99f2","debate_id":"https:\/\/data.oireachtas.ie\/akn\/ie\/debateRecord\/dail\/2025-12-02\/debate\/main","debate_section_id":"https:\/\/data.oireachtas.ie\/akn\/ie\/debateRecord\/dail\/2025-12-02\/debate\/dbsect_3","debate_date":"2025-12-02","speech_order":"3","speaker_ref":"#MarkDaly","speaker_name":"An Cathaoirleach","speaker_member_code":"Mark-Daly.S.2007-07-23","speaker_match_method":"xml_tlc_person_href","speaker_match_confidence":"1.0","speech_text":"A Uachtar\u00e1n Zelenskyy, a Cheann Comhairle agus a dhaoine uaisle, President Zelenskyy, Ceann Comhairle and distinguished guests, on behalf of Members of both Houses and on my own behalf, I thank you for being here today at the heart of our democracy, and for the address delivered to this joint sitting of D\u00e1il \u00c9ireann and Seanad \u00c9ireann. In 2022, when I had the privilege of giving the closing remarks to your online address to these Houses, I said that as we sat here in Dublin |
| silver_division_tallies | official_api_sample_match | all selected stable fields equal | mismatch | division_tally:7201ccfb841a77e91d3fcc06 | {"member_count": {"live": "0", "official": ""}} |
| silver_division_tallies | official_api_sample_match | all selected stable fields equal | mismatch | division_tally:40d24d87be8c00094ca2cd0d | {"member_count": {"live": "0", "official": ""}} |
| silver_division_tallies | official_api_sample_match | all selected stable fields equal | mismatch | division_tally:7f108f7fd992f443ef2d4309 | {"member_count": {"live": "0", "official": ""}} |
| silver_division_tallies | official_api_sample_match | all selected stable fields equal | mismatch | division_tally:b94d4ddb18cb92ad06d6a9cd | {"member_count": {"live": "0", "official": ""}} |
