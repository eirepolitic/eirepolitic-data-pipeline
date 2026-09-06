#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, re, unicodedata
import boto3, pandas as pd
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3')
def read(key):
    o=s3.get_object(Bucket=BUCKET,Key=key); return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)
def norm(s):
    s=unicodedata.normalize('NFKD',str(s)); s=''.join(c for c in s if not unicodedata.combining(c)); s=s.lower(); s=re.sub(r'[^a-z0-9]+',' ',s); return ' '.join(s.split())
sp=read('processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv')
cm=read('processed/oireachtas_unified/latest/csv/gold_current_members.csv')
print('SPEECH_COLS',json.dumps(list(sp.columns),ensure_ascii=False)); print('CURRENT_MEMBER_COLS',json.dumps(list(cm.columns),ensure_ascii=False)); print('CURRENT_MEMBER_ROWS',len(cm))
sp['_date']=pd.to_datetime(sp['Debate Date'],errors='coerce'); sp['_name_norm']=sp['Speaker Name'].map(norm); sp['_words']=sp['Speech Text'].fillna('').astype(str).str.findall(r"\b\w+[\w’'\-]*\b").str.len()
name_col=next((c for c in ['full_name','member_name','name','display_name'] if c in cm.columns),None)
code_col=next((c for c in ['member_code','memberCode','member_id'] if c in cm.columns),None)
party_col=next((c for c in ['party_name','party','current_party'] if c in cm.columns),None)
const_col=next((c for c in ['constituency_name','constituency','current_constituency'] if c in cm.columns),None)
start_col=next((c for c in ['membership_start_date','start_date','date_from'] if c in cm.columns),None)
print('RESOLVED_MEMBER_COLS',json.dumps({'name':name_col,'code':code_col,'party':party_col,'constituency':const_col,'start':start_col}))
cm['_name_norm']=cm[name_col].map(norm)
# only unique name matches
uniq=cm.groupby('_name_norm').filter(lambda x: len(x)==1).drop_duplicates('_name_norm')
keep=['_name_norm',name_col]+[c for c in [code_col,party_col,const_col,start_col] if c]
w=sp[sp['_date']>=pd.Timestamp('2024-12-18')].merge(uniq[keep],on='_name_norm',how='inner')
print('SESSION_RANGE',str(w['_date'].min().date()),str(w['_date'].max().date()),'DEBATE_DAYS',int(w['_date'].dt.date.nunique()),'TD_SPEECH_ROWS',len(w),'MATCHED_TDS',w[name_col].nunique())
# Aggregate TD speech metrics.
g=w.groupby([name_col]+[c for c in [party_col,const_col,start_col] if c],dropna=False).agg(speech_count=('_words','size'),total_words=('_words','sum'),avg_words=('_words','mean'),median_words=('_words','median'),longest_speech_words=('_words','max'),speaking_days=('_date',lambda x:x.dt.date.nunique())).reset_index()
def emit(label,df): print(label,json.dumps(df.to_dict('records'),ensure_ascii=False))
emit('MOST_SPEECHES',g.sort_values(['speech_count','total_words'],ascending=False).head(10))
emit('MOST_TOTAL_WORDS',g.sort_values(['total_words','speech_count'],ascending=False).head(10))
emit('LONGEST_AVG_MIN20',g[g.speech_count>=20].sort_values(['avg_words','speech_count'],ascending=False).head(10))
emit('LONGEST_SINGLE_SPEECH',g.sort_values(['longest_speech_words','total_words'],ascending=False).head(10))
emit('MOST_SPEAKING_DAYS',g.sort_values(['speaking_days','speech_count'],ascending=False).head(10))
# Fair low-end comparison: members whose current membership began no later than the first covered debate day, if start date exists.
if start_col:
    g['_start']=pd.to_datetime(g[start_col],errors='coerce')
    eligible=g[(g['_start'].isna())|(g['_start']<=w['_date'].min())].copy()
else: eligible=g.copy()
emit('FEWEST_SPEECHES_FULL_WINDOW',eligible.sort_values(['speech_count','total_words']).head(15))
# Locate the actual longest speech rows for headline verification.
mx=w['_words'].max(); cols=[c for c in [name_col,party_col,const_col,'Debate Date','Debate Section','Debate Section Name','Speech Order'] if c in w.columns]
emit('LONGEST_SPEECH_ROWS',w[w['_words']==mx][cols+['_words']].head(10))
