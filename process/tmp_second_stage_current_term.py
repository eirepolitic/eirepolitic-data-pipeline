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
    rk=resolve_production_key(s3,bucket=BUCKET,production_key=k)
    obj=s3.get_object(Bucket=BUCKET,Key=rk)
    return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False),rk
f={}; resolved={}
for n,k in KEYS.items(): f[n],resolved[n]=read(k)
snap=build_bill_content_snapshot(bills=f['bills'],stages=f['stages'],sponsors=f['sponsors'],bill_debate_sections=f['bridge'],speeches=f['speeches'],divisions=f['divisions'],member_votes=f['member_votes'],batch_size=6)
second=snap[snap['series_bucket'].eq('second_stage')].copy()
second['_last']=pd.to_datetime(second['last_event_date'],errors='coerce')
# Current parliamentary term thresholds: 34th Dail first sat 2024-12-18; 27th Seanad first sat 2025-02-12.
def threshold(row):
    house=str(row.get('current_stage_house_name') or row.get('origin_house_name') or '').lower()
    return pd.Timestamp('2025-02-12') if 'seanad' in house else pd.Timestamp('2024-12-18')
second['_current_term_active']=second.apply(lambda r: pd.notna(r['_last']) and r['_last']>=threshold(r),axis=1)
active=second[second['_current_term_active']].copy().sort_values(['_last','bill_year','bill_no'],ascending=[False,False,False])
cols=['bill_id','bill_no','bill_year','title','origin_house_name','current_stage_house_name','introduced_date','last_event_date','current_stage_date','primary_sponsor_name','certified_section_count','certified_speech_count','certified_division_count','latest_division_subject','latest_division_outcome']
rows=active[[c for c in cols if c in active.columns]].to_dict(orient='records')
out={'resolved_keys':resolved,'total_currently_second_stage':int(len(second)),'current_term_active_currently_second_stage':int(len(active)),'rows':rows}
Path('artifacts/second-stage-current-term').mkdir(parents=True,exist_ok=True)
Path('artifacts/second-stage-current-term/summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'total_currently_second_stage':len(second),'current_term_active_currently_second_stage':len(active),'titles':[r['title'] for r in rows]},ensure_ascii=False,indent=2))
