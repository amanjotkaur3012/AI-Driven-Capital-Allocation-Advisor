def apply_scenario(revenue, cost, scenario):
    """
    Applies macroeconomic scenario shocks.

    Best Case:
    - Revenue +15%

    Worst Case:
    - Revenue -20%
    - Cost +10%
    """
    if scenario == "Best":
        revenue *= 1.15
    elif scenario == "Worst":
        revenue *= 0.80
        cost *= 1.10

    return revenue, cost
