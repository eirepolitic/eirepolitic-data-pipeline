#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))

import boto3, pandas as pd
from extract.oireachtas.batch import resolve_production_key

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3')
KEYS={
 'member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv',
 'member_parties':'processed/oireachtas_unified/latest/csv/silver_member_parties.csv',
 'memberships':'processed/oireachtas_unified/latest/csv/silver_member_memberships.csv',
}

def read(logical):
    key=resolve_production_key(s3,bucket=BUCKET,production_key=logical)
    o=s3.get_object(Bucket=BUCKET,Key=key)
    return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)

f={k:read(v) for k,v in KEYS.items()}
for n,d in f.items(): print('TABLE',n,len(d),list(d.columns))

vote_id='https://data.oireachtas.ie/ie/oireachtas/division/house/dail/34/2026-07-07/vote_169'
v=f['member_votes'][f['member_votes']['division_id']==vote_id].copy()
assert not v.empty, vote_id
vote_date=pd.Timestamp('2026-07-07')

def kind(x):
    x=str(x or '').strip().casefold()
    if x in ['yes','ta','tá','aye','for']: return 'for'
    if x in ['no','nil','níl','noe','against']: return 'against'
    if 'abst' in x or 'staon' in x: return 'abstain'
    return 'other'
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
amb=p_active.groupby('member_code')['party_name'].nunique().sort_values(ascending=False)
print('AMBIGUOUS_PARTY_MEMBERS',json.dumps(amb[amb>1].to_dict(),ensure_ascii=False))
p_one=p_active.sort_values(['member_code','_start','party_name']).drop_duplicates('member_code',keep='last')[['member_code','party_name']]

elig=active[['member_code']].merge(p_one,on='member_code',how='left')
elig['party_name']=elig['party_name'].fillna('').str.strip().replace('', 'Independent / no party')

v2=v[['member_code','member_name','vote_kind']].merge(p_one,on='member_code',how='left')
v2['party_name']=v2['party_name'].fillna('').str.strip().replace('', 'Independent / no party')

print('RECORDED_COUNTS',json.dumps(v2['vote_kind'].value_counts().to_dict(),ensure_ascii=False))
print('ELIGIBLE_MEMBER_COUNT',len(elig))
print('RECORDED_MEMBER_COUNT',v2['member_code'].nunique())
print('NO_RECORDED_VOTE_COUNT',len(set(elig['member_code'])-set(v2['member_code'])))
print('VOTERS_NOT_IN_ELIGIBLE_SET',json.dumps(sorted(set(v2['member_code'])-set(elig['member_code'])),ensure_ascii=False))

overall={k:int((v2['vote_kind']==k).sum()) for k in ['for','against','abstain','other']}
overall['no_recorded_vote']=int(len(set(elig['member_code'])-set(v2['member_code']))); overall['eligible']=int(len(elig))
print('OVERALL',json.dumps(overall,ensure_ascii=False))

rows=[]
for party,eg in elig.groupby('party_name'):
    codes=set(eg['member_code']); vg=v2[v2['member_code'].isin(codes)]
    row={'party':party,'eligible':len(codes)}
    for k in ['for','against','abstain','other']: row[k]=int((vg['vote_kind']==k).sum())
    row['no_recorded_vote']=len(codes)-vg['member_code'].nunique()
    row['recorded']=vg['member_code'].nunique()
    rows.append(row)
rows=sorted(rows,key=lambda r:(-r['eligible'],r['party']))
print('PARTY_ROWS',json.dumps(rows,ensure_ascii=False,indent=2))
