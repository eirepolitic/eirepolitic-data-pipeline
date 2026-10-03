#!/usr/bin/env python3
from __future__ import annotations
import io,json,os,sys
from pathlib import Path
import boto3,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extract.oireachtas.batch import resolve_production_key
from political_metrics.bill_content_snapshot import build_bill_content_snapshot
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
KEYS={'bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv','stages':'processed/oireachtas_unified/latest/csv/silver_bill_stages.csv','sponsors':'processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv','bridge':'processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv','speeches':'processed/oireachtas_unified/latest/csv/silver_speeches.csv','divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv','member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv'}
s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1'))
def read(k):
 rk=resolve_production_key(s3,bucket=BUCKET,production_key=k); o=s3.get_object(Bucket=BUCKET,Key=rk); return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False),rk
f={};resolved={}
for n,k in KEYS.items(): f[n],resolved[n]=read(k)
s=build_bill_content_snapshot(bills=f['bills'],stages=f['stages'],sponsors=f['sponsors'],bill_debate_sections=f['bridge'],speeches=f['speeches'],divisions=f['divisions'],member_votes=f['member_votes'],batch_size=6)
x=s[s['series_bucket'].eq('second_stage')].copy(); x['sort_date']=pd.to_datetime(x['current_stage_date'],errors='coerce')
x=x.sort_values(['sort_date','bill_year','bill_no'],ascending=[False,False,False])
cols=['bill_id','bill_no','bill_year','title','origin_house_name','bill_type','introduced_date','current_stage_date','current_stage_house_name','current_stage_outcome','primary_sponsor_name','certified_section_count','certified_speech_count','certified_division_count','latest_division_subject','latest_division_date','latest_division_outcome','latest_ta','latest_nil','latest_staon','vote_breakdown_status']
top=x[[c for c in cols if c in x.columns]].head(20)
def num(col): return pd.to_numeric(x[col],errors='coerce').fillna(0) if col in x.columns else pd.Series([0]*len(x),index=x.index)
summary={'resolved_keys':resolved,'count':int(len(x)),'with_debate':int((num('certified_section_count')>0).sum()),'with_speeches':int((num('certified_speech_count')>0).sum()),'with_division':int((num('certified_division_count')>0).sum()),'top20':top.to_dict(orient='records')}
Path('artifacts/second-stage-compact').mkdir(parents=True,exist_ok=True)
Path('artifacts/second-stage-compact/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
