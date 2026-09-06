#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
import boto3, pandas as pd

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
s3=boto3.client('s3')

def read(key):
    obj=s3.get_object(Bucket=BUCKET,Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False)

def pair_counts(df, cols):
    out=[]
    for key,n in df.groupby(cols,dropna=False).size().items():
        if not isinstance(key,tuple): key=(key,)
        out.append({**{c:str(v) for c,v in zip(cols,key)},'count':int(n)})
    return out

b=read('processed/oireachtas_unified/latest/csv/silver_bills.csv')
st=read('processed/oireachtas_unified/latest/csv/silver_bill_stages.csv')
sp=read('processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv')
print('BILL_ROWS',len(b))
print('STATUS_COUNTS',json.dumps(b['status'].value_counts().to_dict(),ensure_ascii=False))

st=st.copy()
st['_date']=pd.to_datetime(st['stage_date'],errors='coerce')
st['_order']=pd.to_numeric(st['order_in_bill'],errors='coerce')
st['_row']=range(len(st))
st=st.sort_values(['bill_id','_date','_order','_row'])
latest=st.groupby('bill_id',as_index=False).tail(1).copy()
latest=latest[['bill_id','stage_name','stage_date','house_name','stage_outcome','order_in_bill']]
x=b.merge(latest,on='bill_id',how='left',suffixes=('','_latest'))
print('ALL_LATEST_STAGE_COUNTS',json.dumps(pair_counts(x,['status','stage_name']),ensure_ascii=False))
current=x[x['status'].str.casefold()=='current'].copy()
print('CURRENT_BILLS',len(current))
print('CURRENT_STAGE_COUNTS',json.dumps(current['stage_name'].value_counts(dropna=False).to_dict(),ensure_ascii=False))
print('CURRENT_STAGE_HOUSE_COUNTS',json.dumps(pair_counts(current,['stage_name','house_name']),ensure_ascii=False))
for status,grp in x.groupby('status',dropna=False):
    print('STATUS_DETAIL',status,len(grp),json.dumps(grp['stage_name'].value_counts(dropna=False).to_dict(),ensure_ascii=False))
cols=['bill_no','bill_year','short_title','title','status','origin_house_name','stage_name','house_name','stage_date','stage_outcome','order_in_bill']
print('CURRENT_BILL_ROWS')
print(json.dumps(current.sort_values(['stage_name','house_name','stage_date','bill_year','bill_no'])[cols].to_dict('records'),ensure_ascii=False,indent=2))
if not sp.empty:
    p=sp.groupby('bill_id').agg(sponsor_rows=('bill_id','size'),sponsor_names=('sponsor_name',lambda s:' | '.join(sorted(set(v for v in s if str(v).strip())))),sponsor_roles=('sponsor_role_name',lambda s:' | '.join(sorted(set(v for v in s if str(v).strip()))))).reset_index()
    c=current[['bill_id','bill_no','bill_year','short_title','stage_name','house_name']].merge(p,on='bill_id',how='left')
    print('CURRENT_SPONSOR_ROW_STATS',json.dumps(pd.to_numeric(c['sponsor_rows'],errors='coerce').describe().round(2).to_dict()))
    print('CURRENT_MULTI_SPONSOR_BILLS',int((pd.to_numeric(c['sponsor_rows'],errors='coerce')>1).sum()))
