# LendingClub Policy Optimization

This repository implements both a **supervised learning model** and an **offline reinforcement learning agent** for loan approval using the LendingClub accepted loans dataset. The goal is to compare risk-based decisioning (supervised) vs reward-based decisioning (RL) for profit maximization.

## Repo Structure

| File | Purpose |
|------|---------|
| `preprocess.py` | Loads raw CSV, cleans data, chooses safe features, encodes + scales, saves train/val/test splits |
| `train_supervised.py` | Trains MLP classifier on processed data |
| `train_rl.py` | Creates offline RL dataset (synthetic denies) and trains a TD3+BC/CQL policy (optional) |
| `evaluate.py` | Evaluates both supervised and RL policies |
| `utils.py` | Helper functions |
| `processed/` | Generated preprocessing artifacts + splits |
| `results/` | Saved models + metrics |

## Dataset

Download `accepted_2007_to_2018Q3.csv.gz` from Kaggle (LendingClub Loan Data)  
and place it in the **project root** (same folder as preprocess.py).

Do NOT unzip — the script handles `.csv.gz` directly.

## Running

1. Put `accepted_2007_to_2018Q3.csv.gz` in repo root  
2. Run preprocessing:
```bash
python preprocess.py
````

3. Train supervised model:

```bash
python train_supervised.py
```

4. (Optional) Train RL:

```bash
python train_rl.py
```

5. Evaluate:

```bash
python evaluate.py
```

## Notes

* The source dataset includes **only approved loans**, so RL training uses **synthetic deny = reward 0** transitions.
* Reward shaping:

  * Approve + fully paid → profit (loan_amnt × int_rate)
  * Approve + default → loss (−loan_amnt)
  * Deny → 0
* Preprocessing artifacts (scaler + features) are saved for reproducibility.

## Next Steps (optional)

* Add doubly-robust OPE
* Tune RL hyperparameters
* Calibrate supervised probabilities

---

This repo demonstrates end-to-end credit decision learning: preprocessing → supervised baseline → offline RL → comparison. Suitable for interview or applied research demonstration.
