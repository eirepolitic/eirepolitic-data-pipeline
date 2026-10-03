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
# Production silver_bill_stages currently exposes bill_id, stage, house and date fields under these possible names.
def pick(cols):
    for c in cols:
        if c in st.columns: return c
    return None
idc=pick(['bill_id','bill_uri','billId','bill_no'])
datec=pick(['stage_date','date','event_date','stageDate','stage_event_date','stage_date_time'])
housec=pick(['house_name','house','stage_house_name','houseName','chamber'])
if not (idc and datec):
    out={'error':'schema_mismatch','stage_columns':list(st.columns),'snapshot_columns':list(snap.columns),'resolved_keys':resolved,'total_currently_second_stage':int(len(second))}
else:
    st['_date']=pd.to_datetime(st[datec],errors='coerce')
    def in_term(row):
        dt=row['_date']
        if pd.isna(dt): return False
        h=str(row[housec]).lower() if housec else ''
        if 'seanad' in h: return dt >= pd.Timestamp('2025-02-12')
        if 'dáil' in h or 'dail' in h: return dt >= pd.Timestamp('2024-12-18')
        return dt >= pd.Timestamp('2024-12-18')
    st['_in_term']=st.apply(in_term,axis=1)
    active_ids=set(st.loc[st['_in_term'],idc].astype(str))
    snap_id='bill_id' if 'bill_id' in second.columns else idc
    active=second[second[snap_id].astype(str).isin(active_ids)].copy()
    active['_sort']=pd.to_datetime(active['last_event_date'],errors='coerce')
    active=active.sort_values(['_sort','bill_year','bill_no'],ascending=[False,False,False])
    cols=['bill_id','bill_no','bill_year','title','origin_house_name','current_stage_house_name','introduced_date','last_event_date','current_stage_date','primary_sponsor_name','certified_section_count','certified_speech_count','certified_division_count','latest_division_subject','latest_division_outcome']
    rows=active[[c for c in cols if c in active.columns]].to_dict(orient='records')
    out={'resolved_keys':resolved,'stage_id_column':idc,'stage_date_column':datec,'stage_house_column':housec,'total_currently_second_stage':int(len(second)),'current_term_active_currently_second_stage':int(len(active)),'rows':rows}
Path('artifacts/second-stage-current-term').mkdir(parents=True,exist_ok=True)
Path('artifacts/second-stage-current-term/summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
