"""
evaluate.py
Evaluate supervised models and produce simple RL policy estimates (Direct Method).
This script tries to be robust: it will attempt to load LightGBM and an MLP model if present.
It also computes a simple Direct-Method OPE for an RL policy using a fitted reward model.

Outputs:
 - results/metrics_lgbm.json (if LGBM present)
 - results/metrics_mlp.json (if MLP present)
 - results/ope_policy_value.json (if RL models/dataset present)
"""
import os, joblib, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

RESULTS = "results"
PROC = "processed"

# load test set
test = pd.read_parquet(os.path.join(PROC, "test.parquet"))
feature_cols = joblib.load(os.path.join(PROC, "feature_cols.joblib"))

# 1) LightGBM metrics (if model exists)
lgbm_path = os.path.join(RESULTS, "lgbm_model.pkl")
if os.path.exists(lgbm_path):
    import joblib
    lgbm = joblib.load(lgbm_path)
    p = lgbm.predict(test[feature_cols])
    auc = float(roc_auc_score(test['y_default'], p))
    f1 = float(f1_score(test['y_default'], (p > 0.5).astype(int)))
    print("LightGBM: test AUC=", auc, "F1=", f1)
    with open(os.path.join(RESULTS, "metrics_lgbm.json"), "w") as f:
        json.dump({"lgbm_auc": auc, "lgbm_f1": f1}, f)
else:
    print("LightGBM model not found at", lgbm_path)

# 2) MLP metrics (if model file exists)
mlp_path = os.path.join(RESULTS, "mlp_best.pt")
mlp_metrics_path = os.path.join(RESULTS, "metrics_mlp.json")
if os.path.exists(mlp_path):
    # try to load a simple MLP (structure must match training file)
    try:
        import torch
        class MLP(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(d,256), torch.nn.ReLU(), torch.nn.BatchNorm1d(256), torch.nn.Dropout(0.2),
                    torch.nn.Linear(256,128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                    torch.nn.Linear(128,1), torch.nn.Sigmoid()
                )
            def forward(self,x): return self.net(x).squeeze(-1)
        model = MLP(len(feature_cols))
        model.load_state_dict(torch.load(mlp_path, map_location='cpu'))
        model.eval()
        X_test = torch.tensor(test[feature_cols].values.astype(np.float32))
        with torch.no_grad():
            preds = model(X_test).numpy()
        auc = float(roc_auc_score(test['y_default'], preds))
        f1 = float(f1_score(test['y_default'], (preds > 0.5).astype(int)))
        print("MLP test AUC=", auc, "F1=", f1)
        with open(mlp_metrics_path, "w") as f:
            json.dump({"mlp_auc": auc, "mlp_f1": f1}, f)
    except Exception as e:
        print("Failed to evaluate MLP (architecture mismatch?):", e)
else:
    print("MLP best checkpoint not found at", mlp_path)

# 3) Simple Direct-Method OPE for RL policy (if RL dataset & policy present)
# We'll fit a simple reward model r_hat(s,a) on the saved RL dataset and evaluate a loaded policy by
# averaging r_hat(s, pi(s)). This is not a low-variance unbiased estimator but is simple and reproducible.
d3rl_npz = os.path.join(RESULTS, "d3rl_dataset.npz")
policy_path = os.path.join(RESULTS, "td3bc_policy")
ope_out = {}

if os.path.exists(d3rl_npz):
    arr = np.load(d3rl_npz)
    obs = arr['observations']
    actions = arr['actions'].reshape(-1)
    rewards = arr['rewards'].reshape(-1)
    # fit simple ridge regression r ~ obs + action
    X = np.hstack([obs, actions.reshape(-1,1)])
    rmodel = Ridge(alpha=1.0)
    rmodel.fit(X, rewards)
    # evaluate each policy if d3rlpy available
    try:
        from d3rlpy.algos.torch.td3 import TD3  # import to check availability
        from d3rlpy.algos import load_model
        # load saved TD3+BC or CQL if available
        if os.path.exists(policy_path):
            from d3rlpy.algos import TD3PlusBC
            pi = TD3PlusBC()
            pi.build_with_dataset(None)  # placeholder, not strictly necessary
            # load model (this gives d3rlpy object on success)
            try:
                pi.load_model(policy_path)
                # apply policy to obs to get actions (pi.predict expects 2D)
                actions_pi = pi.predict(obs)  # shape (N,)
                # compute expected reward under rmodel
                X_pi = np.hstack([obs, np.array(actions_pi).reshape(-1,1)])
                est_rewards = rmodel.predict(X_pi)
                pv = float(est_rewards.mean())  # estimated per-step reward
                ope_out['td3bc_dm_value'] = pv
                print("TD3+BC (Direct-Method) estimated value per decision:", pv)
            except Exception as e:
                print("Failed to load or apply TD3+BC model via d3rlpy:", e)
    except Exception:
        # d3rlpy not installed: fallback to a simple heuristic policy (approve if predicted default prob < 0.2)
        print("d3rlpy not available; using simple rule policy for OPE estimate.")
        # if lgbm exists, use it; else skip
        if os.path.exists(lgbm_path):
            lgbm = joblib.load(lgbm_path)
            probs = lgbm.predict(test[feature_cols])
            # build policy: approve if predicted default prob < 0.2 (note: assimilation; LGBM predict returns prob)
            # careful: original lgbm predict might return continuous score; treat as prob-like
            policy_actions = (probs < 0.2).astype(int)
            # evaluate via rmodel on test obs
            # need obs corresponding to policy; if sizes differ, use obs[:len(test)]
            obs_for_eval = obs[:len(test)]
            act_for_eval = policy_actions[:len(obs_for_eval)]
            X_pi = np.hstack([obs_for_eval, act_for_eval.reshape(-1,1)])
            est_rewards = rmodel.predict(X_pi)
            pv = float(np.mean(est_rewards))
            ope_out['rule_dm_value'] = pv
            print("Rule policy (Direct-Method) estimated value:", pv)
        else:
            print("No LGBM to form fallback policy; skipping DM OPE.")

    # save OPE outputs
    with open(os.path.join(RESULTS, "ope_policy_value.json"), "w") as f:
        json.dump(ope_out, f)
else:
    print("No RL dataset found at", d3rl_npz)