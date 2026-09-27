#!/usr/bin/env python3
import json
from urllib.parse import urlencode
from urllib.request import urlopen, Request

BILLS = {
    'gas_reserve': 'Strategic Gas Reserve',
    'israeli_settlements': 'Israeli Settlements in the Occupied Palestinian Territory',
    'criminal_civil_defence': 'Criminal Law, Civil Law and Defence',
    'housing_tenancies': 'Housing and Residential Tenancies',
    'pharmacy_contraception': 'Contraception Prescribing Service',
    'ai_regulation': 'Regulation of Artificial Intelligence',
}

params = {'date_start':'2026-05-01','date_end':'2026-07-31','limit':'1000'}
url = 'https://api.oireachtas.ie/v1/votes?' + urlencode(params)
req = Request(url, headers={'User-Agent':'EirePolitic vote audit/1.0'})
with urlopen(req, timeout=60) as r:
    payload = json.load(r)

out = {k: [] for k in BILLS}
for row in payload.get('results', []):
    div = row.get('division', {})
    subject = (div.get('subject') or {}).get('showAs') or ''
    debate = (div.get('debate') or {}).get('showAs') or ''
    hay = f'{subject} {debate}'
    for key, needle in BILLS.items():
        if needle.lower() in hay.lower():
            tallies = div.get('tallies') or {}
            def tally(name):
                obj = tallies.get(name) or {}
                return obj.get('tally', 0)
            out[key].append({
                'date': div.get('date'),
                'datetime': div.get('datetime'),
                'vote_id': div.get('voteId'),
                'uri': div.get('uri'),
                'subject': subject,
                'vote_note': div.get('voteNote'),
                'category': div.get('category'),
                'outcome': div.get('outcome'),
                'house': (div.get('house') or {}).get('showAs'),
                'house_code': (div.get('house') or {}).get('houseCode'),
                'debate': debate,
                'debate_section': (div.get('debate') or {}).get('debateSection'),
                'ta': tally('taVotes'),
                'nil': tally('nilVotes'),
                'staon': tally('staonVotes'),
            })

with open('bill-vote-audit.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,sort_keys=True)
print(json.dumps({k: len(v) for k,v in out.items()}, indent=2))
