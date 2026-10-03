#!/usr/bin/env python3
from __future__ import annotations
import io,json,os,sys,traceback
from pathlib import Path
import boto3,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extract.oireachtas.batch import resolve_production_key
from political_metrics.bill_content_snapshot import build_bill_content_snapshot
OUT=Path('artifacts/second-stage-current-term'); OUT.mkdir(parents=True,exist_ok=True)
try:
  BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
  KEYS={'bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv','stages':'processed/oireachtas_unified/latest/csv/silver_bill_stages.csv','sponsors':'processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv','bridge':'processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv','speeches':'processed/oireachtas_unified/latest/csv/silver_speeches.csv','divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv','member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv'}
  s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1'))
  def read(k):
      rk=resolve_production_key(s3,bucket=BUCKET,production_key=k); obj=s3.get_object(Bucket=BUCKET,Key=rk)
      return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False),rk
  f={}; resolved={}
  for n,k in KEYS.items(): f[n],resolved[n]=read(k)
  snap=build_bill_content_snapshot(bills=f['bills'],stages=f['stages'],sponsors=f['sponsors'],bill_debate_sections=f['bridge'],speeches=f['speeches'],divisions=f['divisions'],member_votes=f['member_votes'],batch_size=6)
  second=snap[snap['series_bucket'].eq('second_stage')].copy()
  second['_last']=pd.to_datetime(second['last_event_date'],errors='coerce')
  active=[]
  for _,row in second.iterrows():
      house=str(row['current_stage_house_name']) if 'current_stage_house_name' in second.columns else str(row.get('origin_house_name',''))
      cutoff=pd.Timestamp('2025-02-12') if 'seanad' in house.lower() else pd.Timestamp('2024-12-18')
      dt=row['_last']
      if pd.notna(dt) and dt>=cutoff: active.append(row)
  active_df=pd.DataFrame(active) if active else second.iloc[0:0].copy()
  if len(active_df): active_df=active_df.sort_values(['_last','bill_year','bill_no'],ascending=[False,False,False])
  cols=['bill_id','bill_no','bill_year','title','origin_house_name','current_stage_house_name','introduced_date','last_event_date','current_stage_date','primary_sponsor_name','certified_section_count','certified_speech_count','certified_division_count']
  rows=active_df[[c for c in cols if c in active_df.columns]].to_dict(orient='records')
  out={'ok':True,'resolved_keys':resolved,'total_currently_second_stage':int(len(second)),'current_term_active_currently_second_stage':int(len(active_df)),'rows':rows}
except Exception as e:
  out={'ok':False,'error':repr(e),'traceback':traceback.format_exc()}
OUT.joinpath('summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
