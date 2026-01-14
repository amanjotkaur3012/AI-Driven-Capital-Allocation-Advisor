import numpy as np
import numpy_financial as npf
from scenario_analysis import apply_scenario

def cashflows(base_revenue, base_cost, years, scenario):
    """
    Generates annual project cash flows
    using AI-forecasted revenue and costs.
    """
    growth = 0.04

    revenue, cost = apply_scenario(base_revenue, base_cost, scenario)

    return [
        (revenue - cost) * (1 + growth) ** i
        for i in range(1, years + 1)
    ]

def npv(cf, investment, wacc):
    return npf.npv(wacc, [-investment] + cf)

def irr(cf, investment):
    return npf.irr([-investment] + cf)

def payback(cf, investment):
    cumulative = 0
    for i, c in enumerate(cf):
        cumulative += c
        if cumulative >= investment:
            return i + 1
    return float("inf")

def risk(cf):
    return np.std(cf) / np.mean(cf)
