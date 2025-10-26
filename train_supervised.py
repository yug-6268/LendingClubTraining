"""
Trains LightGBM baseline (fast) and a simple PyTorch MLP.
Saves models and metrics to results/.
"""
import os, joblib, json
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
import lightgbm as lgb
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

OUT_DIR = 'processed'
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)

# load processed
train = pd.read_parquet(os.path.join(OUT_DIR,'train.parquet'))
val = pd.read_parquet(os.path.join(OUT_DIR,'val.parquet'))
test = pd.read_parquet(os.path.join(OUT_DIR,'test.parquet'))
feature_cols = joblib.load(os.path.join(OUT_DIR,'feature_cols.joblib'))

X_train = train[feature_cols]
y_train = train['y_default']
X_val = val[feature_cols]; y_val = val['y_default']
X_test = test[feature_cols]; y_test = test['y_default']

# --- LightGBM baseline ---
print('Training LightGBM...')
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_val, label=y_val)
params = {
    'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
    'num_leaves': 64, 'verbose': -1, 'seed': 42
}
bst = lgb.train(params, train_set, valid_sets=[valid_set], num_boost_round=1000, early_stopping_rounds=50)
# predict
p_test = bst.predict(X_test)
auc = roc_auc_score(y_test, p_test)
f1 = f1_score(y_test, (p_test>0.5).astype(int))
print('LightGBM test AUC', auc, 'F1', f1)
joblib.dump(bst, os.path.join(RESULTS,'lgbm_model.pkl'))

# save metrics
metrics = {'lgbm_auc': float(auc), 'lgbm_f1': float(f1)}
with open(os.path.join(RESULTS,'metrics_lgbm.json'),'w') as f: json.dump(metrics,f)

# --- Simple MLP (PyTorch) ---
print('Training MLP...')
Xtr = torch.tensor(X_train.values, dtype=torch.float32)
Ytr = torch.tensor(y_train.values, dtype=torch.float32)
Xv = torch.tensor(X_val.values, dtype=torch.float32)
Yv = torch.tensor(y_val.values, dtype=torch.float32)
Xt = torch.tensor(X_test.values, dtype=torch.float32)
Yt = torch.tensor(y_test.values, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xtr,Ytr), batch_size=8192, shuffle=True)
val_loader = DataLoader(TensorDataset(Xv,Yv), batch_size=8192)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x).squeeze(-1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP(Xtr.shape[1]).to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

best_auc = 0.0
patience=5; wait=0
for epoch in range(1,51):
    model.train()
    for xb,yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward(); opt.step()
    # val
    model.eval()
    preds=[]; ys=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            xb = xb.to(device)
            p = model(xb).cpu().numpy()
            preds.append(p); ys.append(yb.numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    auc = roc_auc_score(ys, preds)
    print(f'Epoch {epoch} val_auc {auc:.4f}')
    if auc > best_auc:
        best_auc = auc; wait=0
        torch.save(model.state_dict(), os.path.join(RESULTS,'mlp_best.pt'))
    else:
        wait += 1
        if wait >= patience: break

# final test eval
model.load_state_dict(torch.load(os.path.join(RESULTS,'mlp_best.pt')))
model.eval();
with torch.no_grad():
    ptest = model(Xt.to(device)).cpu().numpy()
auc_test = roc_auc_score(Yt.numpy(), ptest)
f1_test = f1_score(Yt.numpy(), (ptest>0.5).astype(int))
print('MLP test AUC', auc_test, 'F1', f1_test)
metrics['mlp_auc']=float(auc_test); metrics['mlp_f1']=float(f1_test)
with open(os.path.join(RESULTS,'metrics_mlp.json'),'w') as f: json.dump(metrics,f)