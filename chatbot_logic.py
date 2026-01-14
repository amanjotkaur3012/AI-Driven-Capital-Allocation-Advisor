def get_predefined_answers(df):
    """
    Returns predefined executive-level questions and answers
    based on final capital allocation output.
    """

    selected = df[df["Decision"].str.contains("Selected")]["Project_ID"].tolist()
    rejected = df[df["Decision"].str.contains("Rejected")]["Project_ID"].tolist()

    highest_risk = df.sort_values("Risk", ascending=False).iloc[0]
    highest_npv = df.sort_values("NPV", ascending=False).iloc[0]

    return {
        "1️⃣ Which projects were selected for funding?":
            (
                "The following projects were selected for funding based on "
                "their strong risk-adjusted returns and efficient capital usage:\n\n"
                + (", ".join(selected) if selected else "No projects were selected.")
            ),

        "2️⃣ Which projects were rejected due to budget constraints?":
            (
                "The following projects were not funded due to capital constraints:\n\n"
                + (", ".join(rejected) if rejected else "No projects were rejected.")
            ),

        "3️⃣ Which project carries the highest risk?":
            (
                f"Project **{highest_risk['Project_ID']}** carries the highest risk, "
                "as indicated by higher cash-flow volatility."
            ),

        "4️⃣ Which project creates the highest value (NPV)?":
            (
                f"Project **{highest_npv['Project_ID']}** generates the highest Net "
                "Present Value, indicating superior long-term value creation."
            ),

        "5️⃣ What is the overall capital allocation recommendation?":
            (
                "The analysis recommends prioritizing projects that deliver "
                "maximum value per unit of risk within the available capital budget, "
                "while deferring lower-ranked projects due to funding limitations."
            )
    }
