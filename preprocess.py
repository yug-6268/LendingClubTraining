"""
Load accepted_2007_to_2018Q3.csv.gz, perform safe cleaning and feature engineering.
Saves: processed/train.parquet, processed/val.parquet, processed/test.parquet
Also saves pickled transformers in processed/.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

RAW_PATH = 'accepted_2007_to_2018Q3.csv.gz'
OUT_DIR = 'processed'
os.makedirs(OUT_DIR, exist_ok=True)

# --- read (use low_memory False to avoid dtype warnings) ---
print('Loading raw csv...')
df = pd.read_csv(RAW_PATH, compression='gzip', low_memory=False)
print('Loaded', df.shape)

# --- target mapping ---
default_set = set(['Charged Off','Default','Does not meet the credit policy. Status:Charged Off'])
# Exclude ambiguous statuses that are censored
exclude_status = set(['Current','Late (31-120 days)','Late (16-30 days)','In Grace Period'])
df = df[df['loan_status'].notnull()]
df = df[~df['loan_status'].isin(exclude_status)].copy()
print('After removing censored statuses:', df.shape)
df['y_default'] = df['loan_status'].isin(default_set).astype(int)

# --- parse numeric fields ---
# int_rate
if df['int_rate'].dtype == object:
    df['int_rate'] = df['int_rate'].astype(str).str.rstrip('%').replace('nan', np.nan).astype(float)
# term
if 'term' in df.columns:
    df['term_months'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
# earliest_cr_line -> credit_age_years
df['issue_d_parsed'] = pd.to_datetime(df['issue_d'], errors='coerce')
if 'earliest_cr_line' in df.columns:
    df['earliest_cr_line_parsed'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
    df['credit_age_years'] = (df['issue_d_parsed'] - df['earliest_cr_line_parsed']).dt.days/365.25
else:
    df['credit_age_years'] = np.nan

# emp_length
def parse_emp_len(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s == 'n/a': return np.nan
    s = s.replace('+','').replace('years','').replace('year','').replace('< 1', '0').strip()
    try:
        return float(s)
    except:
        return np.nan

if 'emp_length' in df.columns:
    df['emp_length_num'] = df['emp_length'].apply(parse_emp_len)

# --- safe feature list (application-time only) ---
candidate = [
    'loan_amnt','term_months','int_rate','installment',
    'grade','sub_grade','emp_length_num','home_ownership','annual_inc',
    'verification_status','purpose','addr_state','dti','delinq_2yrs',
    'inq_last_6mths','open_acc','revol_bal','revol_util','total_acc',
    'credit_age_years'
]
# keep only existing
features = [c for c in candidate if c in df.columns]
print('Using features:', features)

# Drop explicit leakage columns if present
leak_cols = [c for c in df.columns if any(x in c.lower() for x in ['total_pymnt','total_rec','last_pymnt','recover','out_prncp','collection_recovery_fee','next_pymnt','funded_amnt_inv','out_prncp_inv','last_credit_pull','recoveries','settlement'])]
print('Dropping leakage columns (sample):', leak_cols[:20])
df.drop(columns=[c for c in leak_cols if c in df.columns], inplace=True, errors='ignore')

# Subset
keep = features + ['y_default','issue_d_parsed']
proc = df[keep].copy()

# --- impute numerics & encode categories ---
num_cols = proc.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'y_default']
cat_cols = [c for c in features if c not in num_cols]
print('num_cols', num_cols)
print('cat_cols', cat_cols)

num_imputer = SimpleImputer(strategy='median')
proc[num_cols] = num_imputer.fit_transform(proc[num_cols])

# grade -> ordinal
if 'grade' in proc.columns:
    grade_map = {'A':7,'B':6,'C':5,'D':4,'E':3,'F':2,'G':1}
    proc['grade_ord'] = proc['grade'].map(grade_map).fillna(0).astype(int)
    if 'grade' in cat_cols: cat_cols.remove('grade')
    if 'grade' in features: features.remove('grade')

# sub_grade label encode
le_sub = None
if 'sub_grade' in proc.columns:
    le_sub = LabelEncoder()
    proc['sub_grade_le'] = le_sub.fit_transform(proc['sub_grade'].astype(str))
    if 'sub_grade' in cat_cols: cat_cols.remove('sub_grade')
    if 'sub_grade' in features: features.remove('sub_grade')

# One-hot small categorical columns
ohe_cols = [c for c in ['purpose','home_ownership','verification_status','addr_state'] if c in proc.columns]
ohe = None
if ohe_cols:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ohe_arr = ohe.fit_transform(proc[ohe_cols].astype(str))
    ohe_cols_new = list(ohe.get_feature_names_out(ohe_cols))
    proc_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols_new, index=proc.index)
    proc = pd.concat([proc.drop(columns=ohe_cols), proc_ohe], axis=1)
    # remove from features
    for c in ohe_cols:
        if c in features: features.remove(c)

# final feature list
feature_cols = [c for c in proc.columns if c not in ['y_default','issue_d_parsed','grade','sub_grade','emp_length']]
# ensure ordering stable
feature_cols = [c for c in feature_cols if c in df.columns or c in (['grade_ord','sub_grade_le']+ (ohe_cols_new if ohe_cols else []))]

# scale numerics (StandardScaler)
scaler = StandardScaler()
# find numeric features to scale
final_num = proc.select_dtypes(include=[np.number]).columns.tolist()
final_num = [c for c in final_num if c not in ['y_default']]
proc[final_num] = scaler.fit_transform(proc[final_num])

# Save artifacts
joblib.dump(num_imputer, os.path.join(OUT_DIR,'num_imputer.joblib'))
joblib.dump(scaler, os.path.join(OUT_DIR,'scaler.joblib'))
if le_sub is not None:
    joblib.dump(le_sub, os.path.join(OUT_DIR,'le_sub.joblib'))
if ohe is not None:
    joblib.dump(ohe, os.path.join(OUT_DIR,'ohe.joblib'))
joblib.dump(feature_cols, os.path.join(OUT_DIR,'feature_cols.joblib'))

# Save time-splited datasets
proc['issue_year'] = proc['issue_d_parsed'].dt.year
train = proc[proc['issue_year'] <= 2016].copy()
val   = proc[proc['issue_year'] == 2017].copy()
test  = proc[proc['issue_year'] >= 2018].copy()
print('train/val/test shapes', train.shape, val.shape, test.shape)
train.to_parquet(os.path.join(OUT_DIR,'train.parquet'))
val.to_parquet(os.path.join(OUT_DIR,'val.parquet'))
test.to_parquet(os.path.join(OUT_DIR,'test.parquet'))

print('Preprocessing done. Artifacts saved to', OUT_DIR)