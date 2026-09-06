#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
import boto3, pandas as pd

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
s3=boto3.client('s3')

def read(key):
    obj=s3.get_object(Bucket=BUCKET,Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False)

# Show available debate compatibility files so coverage assumptions are explicit.
keys=[]
for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET,Prefix='processed/oireachtas_unified/compat/debates/'):
    keys.extend([x['Key'] for x in page.get('Contents',[])])
print('DEBATE_KEYS',json.dumps(keys,ensure_ascii=False))

sp=read('processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv')
print('ROWS',len(sp)); print('COLS',json.dumps(list(sp.columns),ensure_ascii=False))

# Resolve common source columns.
def first(cands): return next((c for c in cands if c in sp.columns),None)
date_col=first(['Debate Date','debate_date','date','speech_date'])
member_col=first(['member_code','speaker_member_code','Speaker Member Code','memberCode'])
name_col=first(['Speaker Name','speaker_name','member_name','full_name'])
text_col=first(['Speech','speech','speech_text','Text','text','content'])
print('RESOLVED',json.dumps({'date':date_col,'member':member_col,'name':name_col,'text':text_col},ensure_ascii=False))

if date_col:
    sp['_date']=pd.to_datetime(sp[date_col],errors='coerce')
    print('DATE_RANGE',str(sp['_date'].min().date()),str(sp['_date'].max().date()))
    print('ROWS_BY_YEAR',json.dumps(sp['_date'].dt.year.value_counts().sort_index().dropna().astype(int).to_dict()))
    print('DEBATE_DAYS',int(sp['_date'].dt.date.nunique()))

if text_col:
    sp['_words']=sp[text_col].fillna('').astype(str).str.findall(r"\b\w+[\w’'\-]*\b").str.len()
else:
    raise SystemExit('No speech text column found')

# Current Dáil starts 2024-12-18; use whatever covered dates exist from then onward.
w=sp[sp['_date']>=pd.Timestamp('2024-12-18')].copy()
if member_col:
    w['_member']=w[member_col].astype(str).str.strip()
else:
    w['_member']=''
if name_col: w['_name']=w[name_col].astype(str).str.strip()
else: w['_name']=''
w=w[w['_member']!=''].copy()
print('CURRENT_DAIL_ROWS_WITH_MEMBER',len(w),'DATE_RANGE',str(w['_date'].min().date()),str(w['_date'].max().date()),'DAYS',int(w['_date'].dt.date.nunique()))

agg=w.groupby(['_member','_name'],dropna=False).agg(
    speech_count=('_words','size'), total_words=('_words','sum'), avg_words=('_words','mean'), median_words=('_words','median'), longest_speech_words=('_words','max'), speaking_days=('_date',lambda x:x.dt.date.nunique())
).reset_index()

def out(title,df):
    print(title); print(json.dumps(df.to_dict('records'),ensure_ascii=False,indent=2))

out('MOST_SPEECHES',agg.sort_values(['speech_count','total_words'],ascending=False).head(15))
out('MOST_TOTAL_WORDS',agg.sort_values(['total_words','speech_count'],ascending=False).head(15))
out('LONGEST_AVG_MIN20',agg[agg.speech_count>=20].sort_values(['avg_words','speech_count'],ascending=False).head(15))
out('LONGEST_SINGLE_SPEECH',agg.sort_values(['longest_speech_words','total_words'],ascending=False).head(15))
out('MOST_SPEAKING_DAYS',agg.sort_values(['speaking_days','speech_count'],ascending=False).head(15))
out('FEWEST_SPEECHES_RAW',agg.sort_values(['speech_count','total_words'],ascending=True).head(20))

# Coverage diagnostic for membership history, used to make a fair low-end comparison.
try:
    mem=read('processed/oireachtas_unified/latest/csv/silver_member_memberships.csv')
    print('MEMBERSHIP_COLS',json.dumps(list(mem.columns),ensure_ascii=False))
    print('MEMBERSHIP_ROWS',len(mem))
except Exception as e:
    print('MEMBERSHIP_LOAD_ERROR',type(e).__name__,str(e)[:200])
