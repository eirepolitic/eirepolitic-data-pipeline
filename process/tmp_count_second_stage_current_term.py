#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, sys, traceback
from pathlib import Path
import boto3, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extract.oireachtas.batch import resolve_production_key
from political_metrics.bill_content_snapshot import build_bill_content_snapshot
OUT=Path('artifacts/second-stage-count'); OUT.mkdir(parents=True,exist_ok=True)
try:
    BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
    KEYS={'bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv','stages':'processed/oireachtas_unified/latest/csv/silver_bill_stages.csv','sponsors':'processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv','bridge':'processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv','speeches':'processed/oireachtas_unified/latest/csv/silver_speeches.csv','divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv','member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv'}
    s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1'))
    def read(k):
        rk=resolve_production_key(s3,bucket=BUCKET,production_key=k)
        o=s3.get_object(Bucket=BUCKET,Key=rk)
        return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False),rk
    f={}; resolved={}
    for n,k in KEYS.items(): f[n],resolved[n]=read(k)
    s=build_bill_content_snapshot(bills=f['bills'],stages=f['stages'],sponsors=f['sponsors'],bill_debate_sections=f['bridge'],speeches=f['speeches'],divisions=f['divisions'],member_votes=f['member_votes'],batch_size=6)
    x=s[s['series_bucket'].eq('second_stage')].copy()
    x['_last']=pd.to_datetime(x['last_event_date'],errors='coerce')
    def is_current_term(r):
        house=''
        if 'current_stage_house_name' in x.columns and str(r['current_stage_house_name']).strip(): house=str(r['current_stage_house_name'])
        elif 'origin_house_name' in x.columns: house=str(r['origin_house_name'])
        cutoff=pd.Timestamp('2025-02-12') if 'seanad' in house.lower() else pd.Timestamp('2024-12-18')
        return pd.notna(r['_last']) and r['_last']>=cutoff
    x=x[x.apply(is_current_term,axis=1)].copy().sort_values(['_last','bill_year','bill_no'],ascending=[False,False,False])
    count=len(x)
    q,rem=divmod(count,3)
    sizes=[3]*q
    if rem==1:
        if sizes: sizes[-1]=4
        else: sizes=[1]
    elif rem==2:
        sizes.append(2)
    bills=[]
    for _,r in x.iterrows():
        bills.append({'title':str(r.get('title','')),'last_event_date':str(r.get('last_event_date','')),'house':str(r.get('current_stage_house_name','')),'bill_no':str(r.get('bill_no','')),'bill_year':str(r.get('bill_year',''))})
    out={'ok':True,'production_key':resolved['bills'],'count':count,'post_sizes':sizes,'post_count':len(sizes),'bills':bills}
except Exception as e:
    out={'ok':False,'error':repr(e),'traceback':traceback.format_exc()}
OUT.joinpath('summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
