#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
import boto3, pandas as pd

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
s3=boto3.client('s3')

def read(key):
    obj=s3.get_object(Bucket=BUCKET,Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False)

mv=read('processed/oireachtas_unified/latest/csv/silver_member_votes.csv')
dv=read('processed/oireachtas_unified/latest/csv/silver_divisions.csv')

target_dates={'2026-06-10','2026-07-08'}
target_dv=dv[dv['division_date'].isin(target_dates)].copy()
print('TARGET_DIVISIONS')
print(target_dv.to_json(orient='records',force_ascii=False,indent=2))

for _,d in target_dv.iterrows():
    did=d['division_id']
    sub=mv[mv['division_id']==did].copy()
    if sub.empty: continue
    print('\nDIVISION',did,d.get('division_date',''),d.get('subject',''))
    print('COUNTS',json.dumps(sub['vote_label'].value_counts().to_dict(),ensure_ascii=False))
    party=(sub.groupby(['party_name_at_vote','vote_label']).size().unstack(fill_value=0).reset_index())
    for c in ['yes','no','abstain']:
        if c not in party.columns: party[c]=0
    party['total']=party[['yes','no','abstain']].sum(axis=1)
    party=party.sort_values(['total','party_name_at_vote'],ascending=[False,True])
    print('PARTY_BREAKDOWN')
    print(party.to_json(orient='records',force_ascii=False,indent=2))
    split=[]
    for p,g in sub.groupby('party_name_at_vote'):
        vals=g['vote_label'].value_counts().to_dict()
        nonzero=[k for k,v in vals.items() if v]
        if len(nonzero)>1:
            split.append({'party':p,'counts':vals,'members':g[['member_name','vote_label']].sort_values(['vote_label','member_name']).to_dict('records')})
    print('SPLIT_PARTIES')
    print(json.dumps(split,ensure_ascii=False,indent=2))
    print('MEMBERS')
    print(sub[['member_name','party_name_at_vote','constituency_name_at_vote','vote_label']].sort_values(['vote_label','party_name_at_vote','member_name']).to_json(orient='records',force_ascii=False,indent=2))
