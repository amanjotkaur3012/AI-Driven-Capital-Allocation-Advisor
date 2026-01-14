def priority_weight(priority):
    return {"High": 1.0, "Medium": 0.8, "Low": 0.6}.get(priority, 0.7)
