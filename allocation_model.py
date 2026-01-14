from sklearn.preprocessing import MinMaxScaler

def score_projects(df):
    """
    Weighted multi-criteria scoring model
    incorporating return, risk, and payback.
    """
    scaler = MinMaxScaler()
    df[["NPV_n","IRR_n","Payback_n","Risk_n"]] = scaler.fit_transform(
        df[["NPV","IRR","Payback","Risk"]]
    )

    df["Score"] = (
        0.40 * df["NPV_n"] +
        0.25 * df["IRR_n"] -
        0.20 * df["Risk_n"] +
        0.15 * (1 - df["Payback_n"])
    )

    # Strategic risk penalty
    df["Score"] *= (1 - 0.2 * df["Risk"])

    return df

def allocate(df, budget=100):
    """
    Capital allocation under a fixed budget constraint.
    """
    spent = 0
    selected = []

    for _, row in df.sort_values("Score", ascending=False).iterrows():
        if spent + row["Investment"] <= budget:
            spent += row["Investment"]
            selected.append(row["Project_ID"])

    df["Decision"] = df["Project_ID"].apply(
        lambda x: "Selected (Funded)" if x in selected else "Rejected (Budget Constraint)"
    )

    return df, spent
