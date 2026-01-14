from sklearn.preprocessing import MinMaxScaler

def score_projects(df):
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
    return df

def allocate_budget(df, budget=100):
    spent = 0
    selected = []

    for _, row in df.sort_values("Score", ascending=False).iterrows():
        if spent + row["Initial_Investment"] <= budget:
            spent += row["Initial_Investment"]
            selected.append(row["Project_ID"])

    df["Decision"] = df["Project_ID"].apply(
        lambda x: "Selected" if x in selected else "Rejected"
    )
    return df
