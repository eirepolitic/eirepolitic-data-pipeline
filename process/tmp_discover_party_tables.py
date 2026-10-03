#!/usr/bin/env python3
import boto3, json, os, io, pandas as pd
from extract.oireachtas.batch import resolve_production_key
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1'))
key='processed/oireachtas_unified/latest/csv/silver_member_votes.csv'
resolved=resolve_production_key(s3,bucket=BUCKET,production_key=key)
batch_root=resolved.split('/tables/')[0]+'/'
keys=[]; token=None
while True:
 kw={'Bucket':BUCKET,'Prefix':batch_root+'tables/'}
 if token: kw['ContinuationToken']=token
 r=s3.list_objects_v2(**kw)
 keys += [o['Key'] for o in r.get('Contents',[]) if 'party' in o['Key'].lower() or 'member' in o['Key'].lower()]
 if not r.get('IsTruncated'): break
 token=r.get('NextContinuationToken')
obj=s3.get_object(Bucket=BUCKET,Key=resolved); votes=pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False,nrows=3)
print(json.dumps({'batch_root':batch_root,'candidate_keys':keys,'member_vote_columns':list(votes.columns),'sample_votes':votes.to_dict(orient='records')},ensure_ascii=False,indent=2))
