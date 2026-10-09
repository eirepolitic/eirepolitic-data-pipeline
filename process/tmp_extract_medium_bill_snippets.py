#!/usr/bin/env python3
from __future__ import annotations
import io,json,os,re
from pathlib import Path
import boto3,pandas as pd, requests
from pypdf import PdfReader
from extract.oireachtas.batch import resolve_production_key
TITLES=[
'Adult Safeguarding Strategy Bill 2026','Protection of Voice and Image Bill 2025','Housing Loans Regulations (Fresh Start - Buyout) (Amendment) Bill 2026','Repeal of Exempted Development Regulations Bill 2026','Harassment, Harmful Communications and Related Offences (Amendment) Bill 2026','Education (Leave for Injuries) Bill 2025','Maternity Protection (Child Bereavement) (Amendment) Bill 2026','National Minimum Wage (Inclusion of Young Persons, Apprentices and Interns) Bill 2025','Local Government (Support for Elected Members) Bill 2024','Electricity (Supply) (Amendment) (No. 2) Bill 2025','Emergency Inspection of Dublin Zoo Bill 2025','Domestic Violence (Free Travel Scheme) Bill 2025','Protection of Retail Workers Bill 2025','Health (Scoliosis Treatment Services) Bill 2024','Planning And Development (Exempted Development - External Wall Insulation) Bill 2025','Health (Postponement of Certain Leave) Bill 2024','Domestic Violence (Amendment) (No. 3) Bill 2024']
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1'))
def read(k):
 rk=resolve_production_key(s3,bucket=BUCKET,production_key=k); o=s3.get_object(Bucket=BUCKET,Key=rk); return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)
bills=read('processed/oireachtas_unified/latest/csv/silver_bills.csv'); docs=read('processed/oireachtas_unified/latest/csv/silver_bill_related_docs.csv'); vers=read('processed/oireachtas_unified/latest/csv/silver_bill_versions.csv')
def first_url(row):
 for c in ('format_pdf_url','format_pdf_uri','format_xml_url','format_xml_uri'):
  v=str(row.get(c,'')).strip()
  if v:return v
 return ''
def extract(url):
 r=requests.get(url,timeout=45,headers={'User-Agent':'EirePolitic research pipeline/1.0'}); r.raise_for_status(); reader=PdfReader(io.BytesIO(r.content)); text=' '.join((p.extract_text() or '') for p in reader.pages[:8]); return re.sub(r'\s+',' ',text).strip()
out=[]
for t in TITLES:
 br=bills[bills['title'].eq(t)]
 if not len(br): out.append({'title':t,'error':'bill not found'}); continue
 bid=br.iloc[0]['bill_id']; dd=docs[docs['bill_id'].eq(bid)]; memo=dd[dd['related_doc_label'].str.contains('Explanatory',case=False,na=False)]; kind='Explanatory Memorandum' if len(memo) else 'As Initiated'; url=first_url(memo.iloc[0]) if len(memo) else ''
 if not url:
  vv=vers[(vers['bill_id'].eq(bid)) & (vers['version_label'].str.contains('Initiated',case=False,na=False))]; url=first_url(vv.iloc[0]) if len(vv) else ''
 try: txt=extract(url); out.append({'title':t,'bill_id':bid,'source_kind':kind,'source_url':url,'excerpt':txt[:4500]})
 except Exception as e: out.append({'title':t,'bill_id':bid,'source_kind':kind,'source_url':url,'error':repr(e)})
Path('artifacts/medium-bill-snippets').mkdir(parents=True,exist_ok=True); Path('artifacts/medium-bill-snippets/snippets.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'count':len(out)},indent=2))
