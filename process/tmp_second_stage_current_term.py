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
st=f['stages'].copy()
# Discover IDs/date/house columns from known schema candidates.
def pick(cols):
    for c in cols:
        if c in st.columns: return c
    raise SystemExit(f'none of columns found: {cols}; columns={list(st.columns)}')
idc=pick(['bill_id','bill_uri','billId'])
datec=pick(['stage_date','date','event_date','stageDate'])
housec=pick(['house_name','house','stage_house_name','houseName'])
st['_date']=pd.to_datetime(st[datec],errors='coerce')
# Any stage activity in current parliamentary term. Dail threshold 2024-12-18, Seanad 2025-02-12.
def in_term(row):
    h=str(row[housec]).lower(); dt=row['_date']
    if pd.isna(dt): return False
    if 'seanad' in h: return dt >= pd.Timestamp('2025-02-12')
    if 'dáil' in h or 'dail' in h: return dt >= pd.Timestamp('2024-12-18')
    return dt >= pd.Timestamp('2024-12-18')
st['_in_term']=st.apply(in_term,axis=1)
active_ids=set(st.loc[st['_in_term'],idc].astype(str))
second['_active_current_term']=second['bill_id'].astype(str).isin(active_ids)
active=second[second['_active_current_term']].copy()
active['_sort']=pd.to_datetime(active['last_event_date'],errors='coerce')
active=active.sort_values(['_sort','bill_year','bill_no'],ascending=[False,False,False])
cols=['bill_id','bill_no','bill_year','title','origin_house_name','current_stage_house_name','introduced_date','last_event_date','current_stage_date','primary_sponsor_name','certified_section_count','certified_speech_count','certified_division_count','latest_division_subject','latest_division_outcome']
rows=active[[c for c in cols if c in active.columns]].to_dict(orient='records')
out={'resolved_keys':resolved,'total_currently_second_stage':int(len(second)),'current_term_active_currently_second_stage':int(len(active)),'rows':rows}
Path('artifacts/second-stage-current-term').mkdir(parents=True,exist_ok=True)
Path('artifacts/second-stage-current-term/summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'total_currently_second_stage':len(second),'current_term_active_currently_second_stage':len(active),'titles':[r['title'] for r in rows]},ensure_ascii=False,indent=2))
