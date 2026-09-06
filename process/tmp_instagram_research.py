#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, re, unicodedata
from pathlib import Path
import boto3, pandas as pd
from extract.oireachtas.io_s3 import get_bytes
from extract.oireachtas.batch import PRODUCTION_POINTER_KEY

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3')
def read_csv(key):
    return pd.read_csv(io.BytesIO(get_bytes(s3,bucket=BUCKET,key=key)),dtype=str,keep_default_na=False)
def norm(s):
    s=unicodedata.normalize('NFKD',str(s)); s=''.join(c for c in s if not unicodedata.combining(c)); s=s.lower(); s=re.sub(r'[^a-z0-9]+',' ',s); return ' '.join(s.split())

pointer=json.loads(s3.get_object(Bucket=BUCKET,Key=PRODUCTION_POINTER_KEY)['Body'].read().decode())
members=read_csv('processed/oireachtas_unified/latest/csv/silver_members.csv')
mships=read_csv('processed/oireachtas_unified/latest/csv/silver_member_memberships.csv')
current=read_csv('processed/oireachtas_unified/latest/csv/gold_current_members.csv')
sp=read_csv('processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv')
result={'pointer':pointer,'row_counts':{'silver_members':len(members),'silver_member_memberships':len(mships),'gold_current_members':len(current),'speech_rows':len(sp)},'member_columns':list(members.columns),'membership_columns':list(mships.columns),'speech_columns':list(sp.columns)}

sp['_date']=pd.to_datetime(sp['Debate Date'],errors='coerce'); sp=sp[sp['_date']>=pd.Timestamp('2024-12-18')].copy(); sp['_words']=sp['Speech Text'].fillna('').astype(str).str.findall(r"\b\w+[\w’'\-]*\b").str.len(); debate_days=sorted(set(sp['_date'].dropna().dt.normalize()))
result['coverage']={'start':str(sp['_date'].min().date()),'end':str(sp['_date'].max().date()),'debate_days':len(debate_days),'rows':len(sp)}

if 'member_code' in sp.columns and sp['member_code'].astype(str).str.strip().ne('').any():
    w=sp.copy(); native_mask=w['member_code'].astype(str).str.strip().ne(''); result['native_member_code_rows']=int(native_mask.sum())
    if (~native_mask).any():
        w['_name_norm']=w['Speaker Name'].map(norm); members['_name_norm']=members['full_name'].map(norm); counts=members['_name_norm'].value_counts(); lookup=members[members['_name_norm'].isin(set(counts[counts==1].index))][['_name_norm','member_code']].drop_duplicates('_name_norm'); fallback=w.loc[~native_mask].drop(columns=['member_code']).merge(lookup,on='_name_norm',how='left'); w=pd.concat([w.loc[native_mask],fallback],ignore_index=True,sort=False)
else:
    sp['_name_norm']=sp['Speaker Name'].map(norm); members['_name_norm']=members['full_name'].map(norm); counts=members['_name_norm'].value_counts(); lookup=members[members['_name_norm'].isin(set(counts[counts==1].index))][['_name_norm','member_code']].drop_duplicates('_name_norm'); w=sp.merge(lookup,on='_name_norm',how='left')
w=w[w['member_code'].astype(str).str.strip().ne('')].merge(members[['member_code','full_name','is_current_member']],on='member_code',how='left')

m=mships.copy(); m['_start']=pd.to_datetime(m['membership_start'],errors='coerce'); m['_end']=pd.to_datetime(m['membership_end'],errors='coerce'); mask34=m['house_no'].astype(str).str.strip().eq('34'); m=m[mask34].copy() if mask34.any() else m[m['chamber'].astype(str).str.lower().str.contains('dáil|dail',regex=True)].copy(); period_start=sp['_date'].min().normalize(); period_end=sp['_date'].max().normalize(); m=m[(m['_start'].isna()|(m['_start']<=period_end))&(m['_end'].isna()|(m['_end']>=period_start))].copy()

w=w.merge(m[['member_code','membership_start','membership_end','_start','_end']],on='member_code',how='inner'); w=w[(w['_start'].isna()|(w['_date']>=w['_start']))&(w['_end'].isna()|(w['_date']<=w['_end']+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)))].copy(); dedupe_cols=[c for c in ['Debate Date','Debate ID','Debate Section','Speech Order','member_code'] if c in w.columns]; w=w.sort_values('_date').drop_duplicates(subset=dedupe_cols)

elig=[]
for _, row in m.sort_values(['member_code','_start']).iterrows():
    st=row['_start']; en=row['_end']; eligible=[d for d in debate_days if (pd.isna(st) or d>=st.normalize()) and (pd.isna(en) or d<=en.normalize())]; elig.append({'member_code':row['member_code'],'eligible_debate_days':len(eligible)})
elig=pd.DataFrame(elig).groupby('member_code',as_index=False)['eligible_debate_days'].max()
agg=w.groupby(['member_code','full_name'],dropna=False).agg(intervention_count=('_words','size'),total_words=('_words','sum'),avg_words=('_words','mean'),median_words=('_words','median'),longest_intervention_words=('_words','max'),speaking_days=('_date',lambda x:x.dt.normalize().nunique())).reset_index().merge(elig,on='member_code',how='left'); agg['interventions_per_eligible_day']=agg['intervention_count']/agg['eligible_debate_days'].replace(0,pd.NA); agg['speaking_day_share']=agg['speaking_days']/agg['eligible_debate_days'].replace(0,pd.NA)
member_elig=m[['member_code']].drop_duplicates().merge(members[['member_code','full_name']],on='member_code',how='left').merge(elig,on='member_code',how='left'); allagg=member_elig.merge(agg,on=['member_code','full_name','eligible_debate_days'],how='left')
for c in ['intervention_count','total_words','speaking_days']: allagg[c]=pd.to_numeric(allagg[c],errors='coerce').fillna(0)
allagg['interventions_per_eligible_day']=allagg['intervention_count']/allagg['eligible_debate_days'].replace(0,pd.NA); allagg['speaking_day_share']=allagg['speaking_days']/allagg['eligible_debate_days'].replace(0,pd.NA)
result['matching']={'dail_members_overlapping_period':int(m['member_code'].nunique()),'matched_event_date_members':int(w['member_code'].nunique()),'matched_interventions':len(w),'all_transcript_rows':len(sp)}
def recs(df,n=10):
    out=df.head(n).copy()
    for c in out.select_dtypes(include=['float']).columns: out[c]=out[c].round(2)
    return out.where(pd.notna(out),None).to_dict('records')
result['rankings']={'most_interventions':recs(agg.sort_values(['intervention_count','total_words'],ascending=False),10),'most_total_words':recs(agg.sort_values(['total_words','intervention_count'],ascending=False),10),'longest_avg_min20':recs(agg[agg['intervention_count']>=20].sort_values(['avg_words','intervention_count'],ascending=False),10),'longest_single':recs(agg.sort_values(['longest_intervention_words','total_words'],ascending=False),10),'most_speaking_days':recs(agg.sort_values(['speaking_days','intervention_count'],ascending=False),10),'highest_speaking_day_share_min50eligible':recs(allagg[allagg['eligible_debate_days']>=50].sort_values(['speaking_day_share','intervention_count'],ascending=False),10),'lowest_intervention_rate_min50eligible':recs(allagg[allagg['eligible_debate_days']>=50].sort_values(['interventions_per_eligible_day','intervention_count'],ascending=True),15)}
cc=members[members['full_name'].str.contains('Catherine Connolly',case=False,na=False)][['member_code','full_name','is_current_member']]
if not cc.empty:
    ccx=cc.merge(mships,on='member_code',how='left'); result['catherine_connolly']=ccx[[c for c in ['member_code','full_name','is_current_member','house_no','chamber','membership_start','membership_end','is_current'] if c in ccx.columns]].to_dict('records')
Path('diagnostics').mkdir(exist_ok=True); out=Path(f"diagnostics/member_speech_metrics_run_{os.getenv('GITHUB_RUN_ID','local')}.json"); out.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8'); print(out)
