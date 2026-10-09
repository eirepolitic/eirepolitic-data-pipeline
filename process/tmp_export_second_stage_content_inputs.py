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
    rk=resolve_production_key(s3,bucket=BUCKET,production_key=k); o=s3.get_object(Bucket=BUCKET,Key=rk)
    return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False),rk
f={}; resolved={}
for n,k in KEYS.items(): f[n],resolved[n]=read(k)
s=build_bill_content_snapshot(bills=f['bills'],stages=f['stages'],sponsors=f['sponsors'],bill_debate_sections=f['bridge'],speeches=f['speeches'],divisions=f['divisions'],member_votes=f['member_votes'],batch_size=6)
x=s[s['series_bucket'].eq('second_stage')].copy(); x['_last']=pd.to_datetime(x['last_event_date'],errors='coerce')
def keep(r):
    h=str(r.get('current_stage_house_name','')).lower(); dt=r['_last']
    if pd.isna(dt): return False
    if h in {'34th dáil','34th dail'}: return dt>=pd.Timestamp('2024-12-18')
    if h=='27th seanad': return dt>=pd.Timestamp('2025-02-12')
    return False
x=x[x.apply(keep,axis=1)].copy().sort_values(['_last','bill_year','bill_no'],ascending=[False,False,False])
# Join selected raw bill columns by bill_id where available.
b=f['bills'].copy(); join='bill_id' if 'bill_id' in b.columns and 'bill_id' in x.columns else None
raw_cols=[c for c in b.columns if any(k in c.lower() for k in ['long','title','description','uri','source','type','status','origin','introduced'])]
if join:
    add=b[[join]+[c for c in raw_cols if c!=join]].drop_duplicates(subset=[join])
    x=x.merge(add,on=join,how='left',suffixes=('','_raw'))
cols=['bill_id','bill_no','bill_year','title','short_title','long_title','long_title_en','bill_type','status','origin_house_name','current_stage_house_name','introduced_date','last_event_date','current_stage_date','primary_sponsor_name','primary_sponsor_role_name','certified_section_count','certified_speech_count','certified_division_count','latest_division_subject','latest_division_outcome']
# add any raw columns that might contain long title/source uri
auto=[c for c in x.columns if any(k in c.lower() for k in ['long_title','source_uri','bill_uri','web_uri','oireachtas'])]
use=[]
for c in cols+auto:
    if c in x.columns and c not in use: use.append(c)
out={'resolved':resolved,'columns':use,'rows':x[use].fillna('').to_dict(orient='records')}
Path('artifacts/second-stage-content-inputs').mkdir(parents=True,exist_ok=True)
Path('artifacts/second-stage-content-inputs/inputs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'count':len(x),'columns':use,'sample':out['rows'][:3]},ensure_ascii=False,indent=2))
