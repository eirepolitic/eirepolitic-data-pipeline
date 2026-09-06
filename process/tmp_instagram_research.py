#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
import boto3, pandas as pd
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3')
def read_csv(key):
    o=s3.get_object(Bucket=BUCKET,Key=key); return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)
def read_json(key):
    o=s3.get_object(Bucket=BUCKET,Key=key); return json.loads(o['Body'].read().decode('utf-8'))
keys={
'members':'processed/oireachtas_unified/latest/csv/silver_members.csv',
'memberships':'processed/oireachtas_unified/latest/csv/silver_member_memberships.csv',
'parties':'processed/oireachtas_unified/latest/csv/silver_member_parties.csv',
'constituencies':'processed/oireachtas_unified/latest/csv/silver_member_constituencies.csv',
'current':'processed/oireachtas_unified/latest/csv/gold_current_members.csv',
}
for n,k in keys.items():
    try:
        d=read_csv(k); print('TABLE',n,'ROWS',len(d),'COLS',json.dumps(list(d.columns)))
        for c in ['is_current_member','is_current','house_no','chamber']:
            if c in d: print(n.upper()+'_'+c.upper(),json.dumps(d[c].replace('',pd.NA).dropna().value_counts().head(30).to_dict()))
    except Exception as e: print('LOAD_ERROR',n,type(e).__name__,str(e)[:250])
for table in ['silver_members','silver_member_memberships','gold_current_members']:
    key=f'processed/oireachtas_unified/review/{table}/latest/manifest.json'
    try:
        m=read_json(key)
        picked={k:m.get(k) for k in ['table','run_id','snapshot_date','params','raw_rows','input_rows','current_membership_rows','output_rows','dq_status'] if k in m}
        print('MANIFEST',table,json.dumps(picked,ensure_ascii=False))
    except Exception as e: print('MANIFEST_ERROR',table,type(e).__name__,str(e)[:250])
