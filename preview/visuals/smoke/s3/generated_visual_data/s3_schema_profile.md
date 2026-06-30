# S3 schema profile summary

Review-only schema snapshot for Instagram visual smoke mappings.

- Created at: `2026-06-30T02:24:28.299973+00:00`
- Profile count: `2`
- Range bytes per file: `262144`
- Sample rows per file: `25`
- Download strategy: S3 prefix range read only; full datasets are not downloaded.
- Publishing: this does not publish, schedule, or approve Instagram content.

## debate_issue_counts_s3

- Config: `instagram/visuals/data_mappings/debate_issue_counts_s3.yml`
- S3 key: `processed/debates/debate_speeches_classified.csv`
- Bucket: `eirepolitic-data`
- Region: `ca-central-1`
- Last modified: `2026-03-28T17:53:52+00:00`
- Content range: `bytes 0-262143/61391799`
- Column count: `8`
- Sample rows inspected: `25`
- Range may be truncated: `True`

### Columns

`Debate Date`, `Debate Section`, `Debate Section Name`, `Speaker Name`, `Speech Text`, `Speech Order`, `speech_id`, `PoliticalIssues`

### Likely numeric columns

`Speech Order`

### Mapping candidate matches

- Transform `issue_category_counts` / `count_by`
  - Label matches: `PoliticalIssues`
  - Value matches: _none_

### Sample column coverage

| Column | Non-empty | Blank | Examples | Top sampled values |
| --- | ---: | ---: | --- | --- |
| `Debate Date` | 25 | 0 | 2024-12-18 | 2024-12-18 (25) |
| `Debate Section` | 25 | 0 | dbsect_7; dbsect_8 | dbsect_7 (22); dbsect_8 (3) |
| `Debate Section Name` | 25 | 0 | Ainmniú Iarrthóirí agus Ceann Comhairle a thoghadh - Selection of Candidate and ; Taoiseach a Ainmniú - Nomination of Taoiseach | Ainmniú Iarrthóirí agus Ceann Comhairle a thoghadh - Selection of Candidate and Election of Ceann Comhairle (22); Taoiseach a Ainmniú - Nomination of Taoiseach (3) |
| `Speaker Name` | 25 | 0 | Cléireach na Dála; John McGuinness; Verona Murphy | Cléireach na Dála (7); Verona Murphy (7); Mary Lou McDonald (3); Simon Harris (2); John McGuinness (1) |
| `Speech Text` | 25 | 0 | The next item of business is the selection of the Ceann Comhairle of Dáil Éirean; A Chléirigh agus a chairde, ar an gcéad dul síos, is mór an onóir é dom gur éiri; Go raibh maith agat, Teachta McGuinness. I now call on Deputy Verona Murphy. | The next item of business is the selection of the Ceann Comhairle of Dáil Éireann – Ceann Comhairle Dháil Éireann a roghnú. I will now proceed to the secret ballot in accordance with Standing Order 6. I must inform the House that, having received and examined nominations for the position of Ceann Comhairle, the following is the list of validly nominated candidates: Deputy John McGuinness, Deputy Verona Murphy, Deputy Seán Ó Fearghaíl and Deputy Aengus Ó Snodaigh. As there is more than one candidate, the candidate who will be proposed for election by the House will be selected by secret ballot. Before proceeding to the secret ballot, I will in alphabetical order call on each candidate to speak on their own behalf, or where the candidate has been nominated by another Member to speak on their behalf, I will call on that Member. Each Deputy will have five minutes. I now call on Deputy John McGuinness. (1); A Chléirigh agus a chairde, ar an gcéad dul síos, is mór an onóir é dom gur éirigh liom ainmniúchán do phost an Cheann Comhairle a fháil. Gabhaim buíochas dóibh siúd a thug an t-ainmniúchán dom. Gabhaim míle buíochas freisin do gach feisire a bhí sásta labhairt nuair a bhí mo chanbhasáil ar siúl. Molaim an sárobair a rinne na Teachtaí Seán Ó Fearghaíl agus Catherine Connolly. Gabhaim buíochas leo. I stand here as a candidate for the position of Ceann Comhairle. I have the greatest respect for that office and for the democratic processes within this House. I know there are two sets of opinions here. There is the view that political parties can nominate and suggest to their members how to vote and, in a way, control the outcome. I have a different view which dates back to 2016, when it was decided that the Ceann Comhairle could be nominated by seven Members and the vote would be conducted by secret ballot. I have served since 1997. I have been Minister of State for trade and commerce, Chairman of the Committee of Public Accounts and, more recently, Chairman of the finance committee. In my work on those committees I found that you achieve best by co-operating with the members and by working as one to achieve an end, whatever that end might be. During the course of the Committee of Public Accounts, many harsh decisions had to be made in respect of how we conduct our business, the election of the Chair and so on. I was there for history to be made when Shane Ross contested against me. Unusually, it fell to the Opposition to fill that Chair, but there was a contest. In working together, that committee saw the resolution to problems that were faced by the likes of Maurice McCabe. The committee saw a resolution, or part of a resolution, to an issue relating to the Grace case, and to the 47 others who were affected by sexual, physical and financial abuse. In more recent times, the finance committee came together to deal with matters such as vulture funds and the tracker mortgage issue, where we were told in the beginning it was 3,000 cases and it turned out to be 50,000-plus cases. All of that work was achieved not by me; it was achieved by all of the members of each of those committees working together to achieve what was best in the interest of the State and in the interest of the individuals concerned. I am saying this to the House today because I believe what is best for the Members of this House is to work together to first ensure you have a Parliament here that will function in your interests. If it functions in your interests, it will, therefore, function in the interests of the people you represent. Through my years of public service, I have seen where the Dáil Chamber can often be sidelined and I have seen where Members can often be sidelined. Those who are in Government parties are often referred to as being part of the Government but actually there are times when the Government can be just the Cabinet and there are times when the Government can be just the leaders of the groups in that coalition. Because of that, I believe this House has to exercise its strength and its caution around all of the changes that have occurred in politics over that time. I ask Members as individuals of this Parliament to consider the nominees before you today, protect the interests of this House and the interest of democracy, and elect the best person you see fit to hold the position of Ceann Comhairle to defend your interests and the interests of the people we represent. The last general election told us that we need to build trust with the citizens of this country. We need to restore that trust. I believe the first step in the restoration of that trust is the election of our Ceann Comhairle to reinforce the changes in our democratic structures in the interests of the people we represent and to ensure we work together to deliver for our citizens, keep the Government accountable to this House, ensure transparency and ensure the Government keeps its people safe. (1); Go raibh maith agat, Teachta McGuinness. I now call on Deputy Verona Murphy. (1); I thank the Clerk. I wish to thank the people of Wexford who once again have put their trust in me to represent them. It is an honour and a privilege to do so. I take this opportunity to thank my colleagues in the Regional Group for the confidence they have placed in me, nominating me for the position of Ceann Comhairle. I pay tribute to the outgoing Ceann Comhairle, in particular for the kindness and guidance he has shown to me and many other Members since his election in 2016. Whatever the outcome of this election, I wish my fellow candidates well. Whoever is elected will have a challenging role. I congratulate all of you, the elected Members of the Thirty-fourth Dáil, on your successful election. The multi-seat constituency electoral dynamic tends to incite an unhealthy competition among colleagues within constituencies, leaving most battle-scarred. Former Wexford politician Avril Doyle once described politics as the last blood sport. From speaking to Members in the past week, that would appear to be a very appropriate description. For those of us who were successful and elected, the wounds will heal quickly. For those unsuccessful candidates who put their names forward and did not succeed, the scars may linger. Either way, they are to be commended on allowing their names to go forward. Chambers such as these are the lifeblood of a functioning democracy. Having succeeded in being elected, we are now the voice of our communities in this forum. We come into the House to make known the fears, concerns and aspirations of our constituents and to legislate accordingly. We communicate on their behalf the policies they want addressed. In putting forward the views of our constituents, we make known the needs of the people to those who govern. In order to ensure that every Member is heard, this House must function smoothly and efficiently. Every Member is equal, irrespective of their political persuasion, political party or grouping. Every Member has a mandate and no one mandate is more important than the other. If elected as Ceann Comhairle, I will uphold these simple principles. Much debate has been heard regarding the turnout in the recent election. In many constituencies the turnout dropped below 50%. Large numbers of people in the electorate feel alienated from the political process. They feel that politicians are removed from the reality of day-to-day life. They feel that this Chamber is a talking shop that achieves nothing. It is incumbent on all of us to change that perception and to make the House more relevant to those who feel excluded. If elected, I want to engage with all Members to explore the options available to us to reform the way we conduct our business so that ordinary people feel their voices are being heard. The constituents who elected us have a right to see a fair and transparent democracy in action every time they observe our parliamentary debates and legislative work. In the previous Dáil I served as sub-chair and deputised for the Ceann Comhairle. In that regard, I believe I carried out my duties with respect and impartiality. Generally, my time in the Chair passed without incident, with the exception of the day a certain Kerry TD decided to serenade me with a poor rendition of a Cork song — figure that. At least his phone was turned off. I believe this experience has prepared me for the position. If the House chooses to elect me as the first female Ceann Comhairle in the State, it will signify a diverse, inclusive and forward-looking Thirty-fourth Dáil. If elected, I will execute the office of Ceann Comhairle of Dáil Éireann without fear or favour. (1); Go raibh maith agat, a Theachta Murphy. I now call Deputy Seán Ó Fearghaíl. (1) |
| `Speech Order` | 25 | 0 | 1; 2; 3 | 1 (1); 2 (1); 3 (1); 4 (1); 5 (1) |
| `speech_id` | 25 | 0 | a64c003170a89959a192983c; 35cb119f368a1638b46c9932; 1a13cc2ac9a120259239170a | a64c003170a89959a192983c (1); 35cb119f368a1638b46c9932 (1); 1a13cc2ac9a120259239170a (1); dc817d748cece627918ff65f (1); 4ae56b45c07089de3d83ac61 (1) |
| `PoliticalIssues` | 25 | 0 | Government Operations; NONE; International Affairs and Foreign Aid | Government Operations (12); NONE (12); International Affairs and Foreign Aid (1) |

## member_party_counts_s3

- Config: `instagram/visuals/data_mappings/member_party_counts_s3.yml`
- S3 key: `raw/members/oireachtas_members_34th_dail.csv`
- Bucket: `eirepolitic-data`
- Region: `ca-central-1`
- Last modified: `2026-06-01T15:37:47+00:00`
- Content range: `bytes 0-28963/28964`
- Column count: `8`
- Sample rows inspected: `25`
- Range may be truncated: `False`

### Columns

`full_name`, `first_name`, `last_name`, `constituency`, `party`, `gender`, `member_code`, `uri`

### Likely numeric columns

_none_

### Mapping candidate matches

- Transform `party_counts` / `count_by`
  - Label matches: `party`
  - Value matches: _none_

### Sample column coverage

| Column | Non-empty | Blank | Examples | Top sampled values |
| --- | ---: | ---: | --- | --- |
| `full_name` | 25 | 0 | Ciarán Ahern; William Aird; Catherine Ardagh | Ciarán Ahern (1); William Aird (1); Catherine Ardagh (1); Ivana Bacik (1); Cathy Bennett (1) |
| `first_name` | 25 | 0 | Ciarán; William; Catherine | Catherine (2); Colm (2); Ciarán (1); William (1); Ivana (1) |
| `last_name` | 25 | 0 | Ahern; Aird; Ardagh | Byrne (3); Brennan (2); Burke (2); Ahern (1); Aird (1) |
| `constituency` | 25 | 0 | Dublin South-West; Laois; Dublin South-Central | Dublin South-West (2); Wicklow-Wexford (2); Louth (2); Laois (1); Dublin South-Central (1) |
| `party` | 25 | 0 | Labour Party; Fine Gael; Fianna Fáil | Fine Gael (9); Fianna Fáil (8); Sinn Féin (4); Labour Party (2); People Before Profit-Solidarity (1) |
| `gender` | 0 | 25 |  |  |
| `member_code` | 25 | 0 | Ciarán-Ahern.D.2024-11-29; William-Aird.D.2024-11-29; Catherine-Ardagh.S.2016-04-25 | Ciarán-Ahern.D.2024-11-29 (1); William-Aird.D.2024-11-29 (1); Catherine-Ardagh.S.2016-04-25 (1); Ivana-Bacik.S.2007-07-23 (1); Cathy-Bennett.D.2024-11-29 (1) |
| `uri` | 25 | 0 | https://data.oireachtas.ie/ie/oireachtas/member/id/Ciarán-Ahern.D.2024-11-29; https://data.oireachtas.ie/ie/oireachtas/member/id/William-Aird.D.2024-11-29; https://data.oireachtas.ie/ie/oireachtas/member/id/Catherine-Ardagh.S.2016-04-25 | https://data.oireachtas.ie/ie/oireachtas/member/id/Ciarán-Ahern.D.2024-11-29 (1); https://data.oireachtas.ie/ie/oireachtas/member/id/William-Aird.D.2024-11-29 (1); https://data.oireachtas.ie/ie/oireachtas/member/id/Catherine-Ardagh.S.2016-04-25 (1); https://data.oireachtas.ie/ie/oireachtas/member/id/Ivana-Bacik.S.2007-07-23 (1); https://data.oireachtas.ie/ie/oireachtas/member/id/Cathy-Bennett.D.2024-11-29 (1) |
