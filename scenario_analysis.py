def scenario_adjustment(revenue, cost, scenario):
    if scenario == "Best":
        return revenue * 1.15, cost
    elif scenario == "Worst":
        return revenue * 0.80, cost * 1.10
    return revenue, cost
