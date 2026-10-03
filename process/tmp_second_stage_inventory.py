#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, sys
from pathlib import Path
import boto3, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extract.oireachtas.batch import resolve_production_key
from political_metrics.bill_content_snapshot import build_bill_content_snapshot, audit_bill_content_snapshot

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
KEYS={
'bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv',
'stages':'processed/oireachtas_unified/latest/csv/silver_bill_stages.csv',
'sponsors':'processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv',
'bridge':'processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv',
'speeches':'processed/oireachtas_unified/latest/csv/silver_speeches.csv',
'divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv',
'member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv',
}
s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1'))
def read(k):
    rk=resolve_production_key(s3,bucket=BUCKET,production_key=k)
    obj=s3.get_object(Bucket=BUCKET,Key=rk)
    return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False),rk
frames={}; resolved={}
for n,k in KEYS.items(): frames[n],resolved[n]=read(k)
snap=build_bill_content_snapshot(bills=frames['bills'],stages=frames['stages'],sponsors=frames['sponsors'],bill_debate_sections=frames['bridge'],speeches=frames['speeches'],divisions=frames['divisions'],member_votes=frames['member_votes'],batch_size=6)
audit=audit_bill_content_snapshot(snap,batch_size=6)
second=snap[snap['series_bucket'].eq('second_stage')].copy()
cols=['bill_id','bill_no','bill_year','title','short_title','origin_house_name','bill_type','status','introduced_date','last_event_date','current_stage_name','current_stage_date','current_stage_house_name','current_stage_outcome','primary_sponsor_name','primary_sponsor_role_name','sponsor_count','certified_section_count','certified_speech_count','certified_division_count','latest_division_subject','latest_division_date','latest_division_outcome','latest_ta','latest_nil','latest_staon','vote_breakdown_status']
out=second[[c for c in cols if c in second.columns]].sort_values(['current_stage_date','bill_year','bill_no'],ascending=[False,False,False])
Path('artifacts/second-stage-inventory').mkdir(parents=True,exist_ok=True)
out.to_csv('artifacts/second-stage-inventory/current_second_stage.csv',index=False)
summary={'audit':audit,'resolved_keys':resolved,'current_second_stage_count':len(out),'rows':out.to_dict(orient='records')}
Path('artifacts/second-stage-inventory/current_second_stage.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'count':len(out),'titles':[r['title'] for r in summary['rows'][:20]]},ensure_ascii=False,indent=2))
