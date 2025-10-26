"""
Create offline RL dataset (synthetic denies) from processed train/val/test and train a simple offline RL policy using d3rlpy.
This script is intentionally simple: it shows how to create transitions and run TD3+BC/CQL.
"""
import os, joblib
import numpy as np, pandas as pd
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import TD3, CQL, TD3PlusBC

OUT_DIR = 'processed'
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)

train = pd.read_parquet(os.path.join(OUT_DIR,'train.parquet'))
feature_cols = joblib.load(os.path.join(OUT_DIR,'feature_cols.joblib'))

# Build transitions: for each row, create approve (a=1) with reward = loan_amnt*int_rate/100 if y_default==0 else -loan_amnt
# and synthetic deny (a=0) with reward 0. MDPDataset expects arrays: observations, actions, rewards, terminals
obs = train[feature_cols].values.astype(np.float32)
# load original unscaled loan amount & int_rate to compute reward: try to reconstruct by inverse-scaling if scaler saved
scaler = joblib.load(os.path.join(OUT_DIR,'scaler.joblib'))
feat_order = joblib.load(os.path.join(OUT_DIR,'feature_cols.joblib'))
# inverse transform to raw units for 'loan_amnt' and 'int_rate' only if present
# For simplicity, assume loan_amnt and int_rate present in feat_order and scaler fit includes them
X_full = scaler.inverse_transform(train[feat_order].values)
X_full_df = pd.DataFrame(X_full, columns=feat_order, index=train.index)
loan_amnt_raw = X_full_df['loan_amnt'].values
int_rate_raw = X_full_df['int_rate'].values

# compute approve reward
is_default = train['y_default'].values
reward_approve = np.where(is_default==0, loan_amnt_raw * (int_rate_raw/100.0), -loan_amnt_raw)

# Build arrays doubling dataset (approve + deny)
n = len(train)
obs_dup = np.vstack([obs, obs])
actions = np.concatenate([np.ones(n, dtype=np.int32), np.zeros(n, dtype=np.int32)])
rewards = np.concatenate([reward_approve.astype(np.float32), np.zeros(n, dtype=np.float32)])
terminals = np.ones(len(obs_dup), dtype=bool)

# create MDPDataset
dataset = MDPDataset(obs_dup, actions.reshape(-1,1), rewards.reshape(-1,1), None, terminals)
# save dataset as npz for d3rlpy
dataset.dump(os.path.join(RESULTS,'d3rl_dataset.npz'))

# Train a simple TD3+BC (fast) - configuration minimal
algo = TD3PlusBC(actor_learning_rate=3e-4, batch_size=256)
# Fit (this can be slow on CPU; adjust n_epochs lower for quick runs)
algo.fit(dataset, n_epochs=5)  # short for demo; increase for final runs
algo.save_model(os.path.join(RESULTS,'td3bc_policy'))

# Also try CQL
algo2 = CQL(batch_size=256)
algo2.fit(dataset, n_epochs=5)
algo2.save_model(os.path.join(RESULTS,'cql_policy'))