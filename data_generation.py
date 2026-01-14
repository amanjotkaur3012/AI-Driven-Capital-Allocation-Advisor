import pandas as pd
import numpy as np

def generate_historical_data():
    return pd.DataFrame({
        "Year": np.arange(2018, 2025),
        "Revenue": [120, 128, 135, 140, 150, 158, 165],
        "Operating_Cost": [70, 74, 78, 80, 85, 88, 92],
        "Inflation": [3.5, 3.7, 4.1, 5.0, 6.2, 5.8, 5.5],
        "Demand_Index": [95, 97, 98, 100, 102, 104, 106]
    })

def generate_project_data():
    return pd.DataFrame({
        "Project_ID": ["P1","P2","P3","P4","P5","P6"],
        "Industry": ["Energy","FinTech","Healthcare","Manufacturing","Retail","AI"],
        "Initial_Investment": [20, 25, 15, 30, 18, 22],
        "Project_Life": [5, 6, 4, 7, 5, 6],
        "Risk_Factor": [0.35, 0.50, 0.25, 0.45, 0.30, 0.55],
        "Strategic_Priority": ["High","High","Medium","Medium","Low","High"]
    })
