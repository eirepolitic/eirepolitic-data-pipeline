#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
import boto3, pandas as pd
from extract.oireachtas.batch import resolve_production_key, PRODUCTION_POINTER_KEY
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3')

def read_direct(key):
    o=s3.get_object(Bucket=BUCKET,Key=key); return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)

def read_resolved(key):
    rk=resolve_production_key(s3,bucket=BUCKET,production_key=key)
    o=s3.get_object(Bucket=BUCKET,Key=rk); return rk,pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)

pointer=json.loads(s3.get_object(Bucket=BUCKET,Key=PRODUCTION_POINTER_KEY)['Body'].read().decode())
print('PRODUCTION_POINTER',json.dumps(pointer,ensure_ascii=False))
for key in [
 'processed/oireachtas_unified/latest/csv/silver_members.csv',
 'processed/oireachtas_unified/latest/csv/silver_member_memberships.csv',
 'processed/oireachtas_unified/latest/csv/gold_current_members.csv',
]:
    try:
        direct=read_direct(key); print('DIRECT',key,'ROWS',len(direct))
    except Exception as e: print('DIRECT_ERROR',key,type(e).__name__,str(e)[:200])
    try:
        rk,res=read_resolved(key); print('RESOLVED',key,'->',rk,'ROWS',len(res),'COLS',json.dumps(list(res.columns)))
        for c in ['is_current_member','is_current','house_no','chamber','membership_start_date','membership_end_date']:
            if c in res: print('VALUES',key,c,json.dumps(res[c].replace('',pd.NA).dropna().value_counts().head(20).to_dict()))
    except Exception as e: print('RESOLVED_ERROR',key,type(e).__name__,str(e)[:200])
