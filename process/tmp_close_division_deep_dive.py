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
rows=[]
for _,d in dv[dv['division_date'].isin(target_dates)].iterrows():
    sub=mv[mv['division_id']==d['division_id']].copy()
    if sub.empty: continue
    party=(sub.groupby(['party_name_at_vote','vote_label']).size().unstack(fill_value=0).reset_index())
    for c in ['yes','no','abstain']:
        if c not in party.columns: party[c]=0
    party['total']=party[['yes','no','abstain']].sum(axis=1)
    splits=[]
    for p,g in sub.groupby('party_name_at_vote'):
        vc=g['vote_label'].value_counts().to_dict()
        if sum(1 for v in vc.values() if v)>1:
            splits.append({'party':p,'counts':vc,'members':g[['member_name','vote_label']].sort_values(['vote_label','member_name']).to_dict('records')})
    rows.append({
        'division_id':d['division_id'],
        'division_date':d.get('division_date',''),
        'subject':d.get('subject',''),
        'outcome':d.get('outcome',''),
        'vote_totals':sub['vote_label'].value_counts().to_dict(),
        'party_breakdown':party.sort_values(['total','party_name_at_vote'],ascending=[False,True]).to_dict('records'),
        'split_parties':splits,
    })
print(json.dumps(rows,ensure_ascii=False,indent=2))
