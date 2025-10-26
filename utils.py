import numpy as np

def monthly_payment(P, annual_rate_pct, n_months):
    r = (annual_rate_pct/100.0)/12.0
    if r <= 0:
        return P / n_months
    return (r * P) / (1 - (1 + r) ** (-n_months))