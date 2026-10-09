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
BASE={'bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv','stages':'processed/oireachtas_unified/latest/csv/silver_bill_stages.csv','sponsors':'processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv','bridge':'processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv','speeches':'processed/oireachtas_unified/latest/csv/silver_speeches.csv','divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv','member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv','docs':'processed/oireachtas_unified/latest/csv/silver_bill_related_docs.csv','versions':'processed/oireachtas_unified/latest/csv/silver_bill_versions.csv'}
s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1'))
def read(k):
 rk=resolve_production_key(s3,bucket=BUCKET,production_key=k); o=s3.get_object(Bucket=BUCKET,Key=rk); return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False),rk
f={};resolved={}
for n,k in BASE.items(): f[n],resolved[n]=read(k)
s=build_bill_content_snapshot(bills=f['bills'],stages=f['stages'],sponsors=f['sponsors'],bill_debate_sections=f['bridge'],speeches=f['speeches'],divisions=f['divisions'],member_votes=f['member_votes'],batch_size=6)
x=s[s['series_bucket'].eq('second_stage')].copy(); x['_last']=pd.to_datetime(x['last_event_date'],errors='coerce')
def keep(r):
 h=str(r.get('current_stage_house_name','')).lower(); dt=r['_last']
 return pd.notna(dt) and ((h in {'34th dáil','34th dail'} and dt>=pd.Timestamp('2024-12-18')) or (h=='27th seanad' and dt>=pd.Timestamp('2025-02-12')))
x=x[x.apply(keep,axis=1)].copy(); ids=set(x['bill_id'])
d=f['docs'][f['docs']['bill_id'].isin(ids)].copy(); v=f['versions'][f['versions']['bill_id'].isin(ids)].copy()
# existence checks only for first preferred source per bill to keep calls bounded
rows=[]
for _,r in x.sort_values(['last_event_date','bill_year','bill_no'],ascending=[False,False,False]).iterrows():
 bid=r['bill_id']; dd=d[d['bill_id'].eq(bid)]; vv=v[v['bill_id'].eq(bid)]
 labels=' | '.join(dd['related_doc_label'].dropna().astype(str).tolist())
 vlabs=' | '.join(vv['version_label'].dropna().astype(str).tolist())
 expl=dd[dd['related_doc_label'].str.contains('Explanatory|Memorandum|Memo',case=False,na=False)]
 init=vv[vv['version_label'].str.contains('initiat',case=False,na=False)]
 key=''
 source_kind=''
 if len(expl):
  er=expl.iloc[0]; key=er.get('s3_pdf_key') or er.get('s3_xml_key') or ''; source_kind='explanatory'
 elif len(init):
  er=init.iloc[0]; key=er.get('s3_pdf_key') or er.get('s3_xml_key') or ''; source_kind='initiated'
 exists=False
 if key:
  try: s3.head_object(Bucket=BUCKET,Key=key); exists=True
  except Exception: exists=False
 rows.append({'bill_id':bid,'title':r['title'],'has_explanatory':bool(len(expl)),'has_initiated':bool(len(init)),'preferred_source_kind':source_kind,'preferred_s3_key':key,'preferred_source_exists':exists,'doc_labels':labels,'version_labels':vlabs})
out={'count':len(rows),'with_explanatory':sum(1 for r in rows if r['has_explanatory']),'with_initiated':sum(1 for r in rows if r['has_initiated']),'preferred_source_exists':sum(1 for r in rows if r['preferred_source_exists']),'resolved':resolved,'rows':rows}
Path('artifacts/second-stage-doc-audit').mkdir(parents=True,exist_ok=True); Path('artifacts/second-stage-doc-audit/summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:out[k] for k in ['count','with_explanatory','with_initiated','preferred_source_exists']},indent=2))
