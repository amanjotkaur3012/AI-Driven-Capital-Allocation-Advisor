def get_predefined_answers(df):
    selected = df[df["Decision"].str.contains("Selected")]["Project_ID"].tolist()
    rejected = df[df["Decision"].str.contains("Rejected")]["Project_ID"].tolist()

    highest_risk = df.sort_values("Risk", ascending=False).iloc[0]
    highest_npv = df.sort_values("NPV", ascending=False).iloc[0]

    return {
        "1️⃣ Which projects were selected for funding?":
            "Selected projects:\n\n" +
            (", ".join(selected) if selected else "None"),

        "2️⃣ Which projects were rejected due to budget constraints?":
            "Rejected projects:\n\n" +
            (", ".join(rejected) if rejected else "None"),

        "3️⃣ Which project carries the highest risk?":
            f"{highest_risk['Project_ID']} has the highest risk based on volatility.",

        "4️⃣ Which project creates the highest value (NPV)?":
            f"{highest_npv['Project_ID']} has the highest Net Present Value.",

        "5️⃣ What is the overall capital allocation recommendation?":
            "Prioritize high NPV, lower-risk projects within the capital budget."
    }
