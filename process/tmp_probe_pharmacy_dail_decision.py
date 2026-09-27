#!/usr/bin/env python3
import json, re
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

TITLE='Health (Provision of Contraception Prescribing Service in Retail Pharmacy Businesses) Bill 2026'
params={'chamber':'dail','date_start':'2026-07-15','date_end':'2026-07-15','limit':'50'}
api='https://api.oireachtas.ie/v1/debates?'+urlencode(params)
with urlopen(Request(api,headers={'User-Agent':'EirePolitic procedure audit/1.0'}),timeout=60) as r:
    payload=json.load(r)

matches=[]
for row in payload.get('results',[]):
    rec=row.get('debateRecord',{})
    for wrapped in rec.get('debateSections',[]):
        sec=wrapped.get('debateSection',{})
        if TITLE.lower() in (sec.get('showAs') or '').lower():
            matches.append(sec)

if not matches:
    raise RuntimeError('No matching debate section found')

out=[]
for sec in matches:
    xml_uri=((sec.get('formats') or {}).get('xml') or {}).get('uri')
    if not xml_uri:
        continue
    with urlopen(Request(xml_uri,headers={'User-Agent':'EirePolitic procedure audit/1.0'}),timeout=60) as r:
        xml_bytes=r.read()
    root=ET.fromstring(xml_bytes)
    text=' '.join(t.strip() for t in root.itertext() if t and t.strip())
    # Keep a focused tail plus keyword hits; final procedural language is normally near the end.
    tail=text[-9000:]
    flags={k: bool(re.search(k,text,re.I)) for k in [
        r'division (?:was )?claimed',r'fewer than ten',r'please rise',r'question put and agreed',r'question declared carried',r'bill is hereby passed',r'passed without a division']}
    out.append({'showAs':sec.get('showAs'),'xml_uri':xml_uri,'flags':flags,'tail':tail})

with open('pharmacy-dail-procedure-audit.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps([x['flags'] for x in out],indent=2))
