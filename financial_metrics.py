import numpy as np
import numpy_financial as npf

WACC = 0.11

def estimate_cashflows(revenue, cost, years):
    growth = 0.04
    return [
        (revenue * (1+growth)**i) - (cost * (1+growth)**i)
        for i in range(1, years + 1)
    ]

def npv(cashflows, investment):
    return npf.npv(WACC, [-investment] + cashflows)

def irr(cashflows, investment):
    return npf.irr([-investment] + cashflows)

def payback(cashflows, investment):
    cumulative = 0
    for i, cf in enumerate(cashflows):
        cumulative += cf
        if cumulative >= investment:
            return i + 1
    return float("inf")

def risk(cashflows):
    return np.std(cashflows) / np.mean(cashflows)
