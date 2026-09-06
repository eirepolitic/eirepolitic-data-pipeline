#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
import boto3, pandas as pd

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
s3=boto3.client('s3')

def read(key):
    obj=s3.get_object(Bucket=BUCKET,Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False)

dv=read('processed/oireachtas_unified/latest/csv/silver_divisions.csv')
mv=read('processed/oireachtas_unified/latest/csv/silver_member_votes.csv')
raw=mv['vote_label'].fillna('').astype(str).str.strip().str.lower()

def kind(x):
    if any(t in x for t in ['staon','abstain']): return 'abstain'
    if x in ['tá','ta','yes','aye','for']: return 'yes'
    if x in ['níl','nil','no','noe','against']: return 'no'
    return 'other'

mv['_kind']=raw.map(kind)
c=mv.groupby(['division_id','_kind']).size().unstack(fill_value=0).reset_index()
for k in ['yes','no','abstain','other']:
    if k not in c: c[k]=0
c['margin']=(c['yes']-c['no']).abs()
cols=['division_id','division_date','chamber','subject','outcome','debate_id','debate_section_id','debate_show_as']
c=c.merge(dv[[x for x in cols if x in dv.columns]],on='division_id',how='left')
close=c[c['margin']<=10].sort_values(['margin','division_date','division_id'])
print('CLOSE_DIVISIONS_COUNT',len(close))
print(json.dumps(close[[x for x in ['division_id','division_date','chamber','subject','outcome','yes','no','abstain','margin','debate_id','debate_section_id','debate_show_as'] if x in close.columns]].to_dict('records'),ensure_ascii=False,indent=2))
