#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
from pathlib import Path
import boto3, pandas as pd
from extract.oireachtas.batch import resolve_production_key, PRODUCTION_POINTER_KEY
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3')

def read_direct(key):
    o=s3.get_object(Bucket=BUCKET,Key=key); return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)

def read_resolved(key):
    rk=resolve_production_key(s3,bucket=BUCKET,production_key=key)
    o=s3.get_object(Bucket=BUCKET,Key=rk); return rk,pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)

pointer=json.loads(s3.get_object(Bucket=BUCKET,Key=PRODUCTION_POINTER_KEY)['Body'].read().decode())
result={'production_pointer':pointer,'tables':{}}
for key in [
 'processed/oireachtas_unified/latest/csv/silver_members.csv',
 'processed/oireachtas_unified/latest/csv/silver_member_memberships.csv',
 'processed/oireachtas_unified/latest/csv/gold_current_members.csv',
]:
    item={}
    try:
        direct=read_direct(key); item['direct_rows']=len(direct)
    except Exception as e: item['direct_error']=f'{type(e).__name__}: {str(e)[:200]}'
    try:
        rk,res=read_resolved(key); item['resolved_key']=rk; item['resolved_rows']=len(res); item['columns']=list(res.columns)
        for c in ['is_current_member','is_current','house_no','chamber','membership_start_date','membership_end_date']:
            if c in res:
                item[f'{c}_counts']={str(k):int(v) for k,v in res[c].replace('',pd.NA).dropna().value_counts().head(20).items()}
    except Exception as e: item['resolved_error']=f'{type(e).__name__}: {str(e)[:200]}'
    result['tables'][key]=item
Path('diagnostics').mkdir(exist_ok=True)
Path('diagnostics/pointer_member_check.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
