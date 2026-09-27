#!/usr/bin/env python3
import json
from urllib.parse import urlencode
from urllib.request import urlopen, Request

params={'date_start':'2026-06-10','date_end':'2026-06-10','limit':'200'}
url='https://api.oireachtas.ie/v1/votes?'+urlencode(params)
req=Request(url,headers={'User-Agent':'EirePolitic vote payload audit/1.0'})
with urlopen(req,timeout=60) as r: payload=json.load(r)
for row in payload.get('results',[]):
    div=row.get('division',{})
    if div.get('voteId')=='vote_131':
        with open('vote-payload-sample.json','w',encoding='utf-8') as f:
            json.dump(row,f,ensure_ascii=False,indent=2,sort_keys=True)
        print('division keys',sorted(div.keys()))
        break
