#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, sys
from pathlib import Path
import boto3, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extract.oireachtas.batch import resolve_production_key
from political_metrics.bill_content_snapshot import build_bill_content_snapshot
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
def current_term(r):
    house=str(r.get('current_stage_house_name','') or r.get('origin_house_name','')).lower()
    cutoff=pd.Timestamp('2025-02-12') if 'seanad' in house else pd.Timestamp('2024-12-18')
    return pd.notna(r['_last']) and r['_last']>=cutoff
x=x[x.apply(current_term,axis=1)].copy().sort_values(['_last','bill_year','bill_no'],ascending=[False,False,False])
count=len(x)
# pack into 3-4 bill posts, minimizing number of posts while avoiding 2-bill tails
post_count=(count+3)//4
base=count//post_count if post_count else 0
rem=count%post_count if post_count else 0
sizes=[base+1]*rem+[base]*(post_count-rem) if post_count else []
# If any post would fall below 3, fall back to mostly 3s with one 4.
if sizes and min(sizes)<3:
    q,r=divmod(count,3); sizes=[3]*q
    if r==1 and sizes: sizes[-1]=4
    elif r==2: sizes.append(2)
rows=[]
for _,r in x.iterrows(): rows.append({'title':r['title'],'last_event_date':r['last_event_date'],'house':r.get('current_stage_house_name',''),'bill_no':r.get('bill_no',''),'bill_year':r.get('bill_year','')})
out={'count':count,'post_count':len(sizes),'post_sizes':sizes,'resolved_bills_key':resolved['bills'],'bills':rows}
Path('artifacts/second-stage-count').mkdir(parents=True,exist_ok=True)
Path('artifacts/second-stage-count/summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'count':count,'post_count':len(sizes),'post_sizes':sizes,'titles':[r['title'] for r in rows]},ensure_ascii=False,indent=2))
