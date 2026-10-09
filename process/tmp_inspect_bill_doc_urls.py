#!/usr/bin/env python3
import io,json,os
from pathlib import Path
import boto3,pandas as pd
from extract.oireachtas.batch import resolve_production_key
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1'))
def read(k):
 rk=resolve_production_key(s3,bucket=BUCKET,production_key=k); o=s3.get_object(Bucket=BUCKET,Key=rk); return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False),rk
docs,_=read('processed/oireachtas_unified/latest/csv/silver_bill_related_docs.csv'); vers,_=read('processed/oireachtas_unified/latest/csv/silver_bill_versions.csv')
bid='https://data.oireachtas.ie/ie/oireachtas/bill/2026/94'
out={'docs_columns':list(docs.columns),'docs_rows':docs[docs['bill_id'].eq(bid)].to_dict(orient='records'),'versions_columns':list(vers.columns),'versions_rows':vers[vers['bill_id'].eq(bid)].to_dict(orient='records')}
Path('artifacts/doc-url-inspect').mkdir(parents=True,exist_ok=True); Path('artifacts/doc-url-inspect/sample.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
