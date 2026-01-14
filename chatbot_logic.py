def explain(project_id, df):
    row = df[df["Project_ID"] == project_id].iloc[0]

    if row["Decision"] == "Rejected":
        return f"{project_id} was rejected due to lower risk-adjusted score."
    return f"{project_id} was selected due to superior financial performance."
