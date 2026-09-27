#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))

import boto3, pandas as pd
from extract.oireachtas.batch import resolve_production_key

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
s3=boto3.client('s3')
KEYS={
 'member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv',
 'member_parties':'processed/oireachtas_unified/latest/csv/silver_member_parties.csv',
 'memberships':'processed/oireachtas_unified/latest/csv/silver_member_memberships.csv',
}
SELECTED={
 'gas_reserve':('2026-06-30','https://data.oireachtas.ie/ie/oireachtas/division/house/dail/34/2026-06-30/vote_162'),
 'israeli_services_amendment':('2026-07-07','https://data.oireachtas.ie/ie/oireachtas/division/house/dail/34/2026-07-07/vote_169'),
 'criminal_civil_defence':('2026-06-10','https://data.oireachtas.ie/ie/oireachtas/division/house/dail/34/2026-06-10/vote_131'),
 'housing_tenancies':('2026-07-08','https://data.oireachtas.ie/ie/oireachtas/division/house/dail/34/2026-07-08/vote_174'),
 'ai_regulation':('2026-06-30','https://data.oireachtas.ie/ie/oireachtas/division/house/dail/34/2026-06-30/vote_167'),
}

def read(logical):
    key=resolve_production_key(s3,bucket=BUCKET,production_key=logical)
    o=s3.get_object(Bucket=BUCKET,Key=key)
    return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)

def kind(x):
    x=str(x or '').strip().casefold()
    if x in ['yes','ta','tá','aye','for']: return 'for'
    if x in ['no','nil','níl','noe','against']: return 'against'
    if 'abst' in x or 'staon' in x: return 'abstain'
    return 'other'

f={k:read(v) for k,v in KEYS.items()}
results={}
for key,(date_s,vote_id) in SELECTED.items():
    vote_date=pd.Timestamp(date_s)
    v=f['member_votes'][f['member_votes']['division_id']==vote_id].copy()
    if v.empty: raise RuntimeError(f'missing vote {vote_id}')
    v['vote_kind']=v['vote_label'].map(kind)

    m=f['memberships'].copy()
    m['_start']=pd.to_datetime(m['membership_start'],errors='coerce')
    m['_end']=pd.to_datetime(m['membership_end'],errors='coerce')
    active=m[(m['house_no'].astype(str)=='34') & (m['_start'].isna() | (m['_start']<=vote_date)) & (m['_end'].isna() | (m['_end']>=vote_date))].copy()
    if 'chamber' in active and active['chamber'].astype(str).str.strip().ne('').any():
        mask=active['chamber'].astype(str).str.casefold().str.contains('dail|dáil')
        if mask.any(): active=active[mask].copy()
    active=active.drop_duplicates('member_code')

    p=f['member_parties'].copy()
    p['_start']=pd.to_datetime(p['party_start'],errors='coerce')
    p['_end']=pd.to_datetime(p['party_end'],errors='coerce')
    p_active=p[(p['_start'].isna() | (p['_start']<=vote_date)) & (p['_end'].isna() | (p['_end']>=vote_date))].copy()
    amb=p_active.groupby('member_code')['party_name'].nunique()
    ambiguous=amb[amb>1].to_dict()
    p_one=p_active.sort_values(['member_code','_start','party_name']).drop_duplicates('member_code',keep='last')[['member_code','party_name']]

    elig=active[['member_code']].merge(p_one,on='member_code',how='left')
    elig['party_name']=elig['party_name'].fillna('').str.strip().replace('', 'Independent')
    v2=v[['member_code','member_name','vote_kind']].merge(p_one,on='member_code',how='left')
    v2['party_name']=v2['party_name'].fillna('').str.strip().replace('', 'Independent')

    rows=[]
    for party,eg in elig.groupby('party_name'):
        codes=set(eg['member_code']); vg=v2[v2['member_code'].isin(codes)]
        row={'party':party,'eligible':len(codes)}
        for k in ['for','against','abstain','other']: row[k]=int((vg['vote_kind']==k).sum())
        row['no_recorded_vote']=len(codes)-vg['member_code'].nunique()
        rows.append(row)
    rows=sorted(rows,key=lambda r:(-r['eligible'],r['party']))
    results[key]={
      'vote_id':vote_id,'date':date_s,'ambiguous_party_members':ambiguous,
      'overall':{
        'eligible':len(elig),
        'for':int((v2['vote_kind']=='for').sum()),
        'against':int((v2['vote_kind']=='against').sum()),
        'abstain':int((v2['vote_kind']=='abstain').sum()),
        'other':int((v2['vote_kind']=='other').sum()),
        'no_recorded_vote':len(set(elig['member_code'])-set(v2['member_code'])),
      },
      'party_rows':rows,
      'voters_not_in_eligible':sorted(set(v2['member_code'])-set(elig['member_code'])),
    }

Path('artifacts').mkdir(exist_ok=True)
with open('artifacts/bill-vote-party-audit.json','w',encoding='utf-8') as f:
    json.dump(results,f,ensure_ascii=False,indent=2,sort_keys=True)
print(json.dumps({k:v['overall'] for k,v in results.items()},ensure_ascii=False,indent=2))
