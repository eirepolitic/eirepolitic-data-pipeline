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
mp=read('processed/oireachtas_unified/latest/csv/silver_member_parties.csv')

TARGETS={
 'https://data.oireachtas.ie/ie/oireachtas/division/house/dail/34/2026-06-10/vote_130',
 'https://data.oireachtas.ie/ie/oireachtas/division/house/dail/34/2026-07-08/vote_184',
}

def party_at(member_code,date):
    g=mp[mp['member_code']==member_code].copy()
    if g.empty:return 'Unknown'
    start=pd.to_datetime(g['party_start'],errors='coerce')
    end=pd.to_datetime(g['party_end'],errors='coerce')
    d=pd.Timestamp(date)
    ok=((start.isna())|(start<=d)) & ((end.isna())|(end>=d))
    g=g[ok]
    vals=[x for x in g['party_name'].astype(str).str.strip().tolist() if x]
    vals=list(dict.fromkeys(vals))
    return vals[0] if len(vals)==1 else (' / '.join(vals) if vals else 'Unknown')

rows=[]
for did in TARGETS:
    drow=dv[dv['division_id']==did].iloc[0]
    date=drow['division_date']
    sub=mv[mv['division_id']==did].copy()
    sub['party_at_vote_derived']=[party_at(code,date) for code in sub['member_code']]
    party=(sub.groupby(['party_at_vote_derived','vote_label']).size().unstack(fill_value=0).reset_index())
    for c in ['yes','no','abstain']:
        if c not in party.columns: party[c]=0
    party['total']=party[['yes','no','abstain']].sum(axis=1)
    splits=[]
    for p,g in sub.groupby('party_at_vote_derived'):
        vc=g['vote_label'].value_counts().to_dict()
        if sum(1 for v in vc.values() if v)>1:
            splits.append({'party':p,'counts':vc,'members':g[['member_name','vote_label']].sort_values(['vote_label','member_name']).to_dict('records')})
    rows.append({
      'division_id':did,'date':date,'debate_show_as':drow.get('debate_show_as',''),'subject':drow.get('subject',''),'outcome':drow.get('outcome',''),
      'vote_totals':sub['vote_label'].value_counts().to_dict(),
      'party_breakdown':party.sort_values(['total','party_at_vote_derived'],ascending=[False,True]).to_dict('records'),
      'split_parties':splits,
      'unknown_party_rows':int((sub['party_at_vote_derived']=='Unknown').sum()),
    })
print(json.dumps(rows,ensure_ascii=False,indent=2))
