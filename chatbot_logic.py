def get_predefined_answers(df):
    """
    Returns predefined executive-level questions and answers
    based on final capital allocation output.
    """

    answers = {
        "1️⃣ Which projects were selected for funding?":
            "The following projects were selected based on strong "
            "risk-adjusted returns and capital efficiency:\n\n"
            + ", ".join(
                df[df["Decision"].str.contains("Selected")]["Project_ID"]
            ),

        "2️⃣ Which projects were rejected due to budget constraints?":
            "The following projects were not funded due to limited "
            "capital availability:\n\n"
            + ", ".join(
                df[df["Decision"].str.contains("Rejected")]["Project_ID"]
            ),

        "3️⃣ Which project carries the highest risk?":
            f"Project **{df.sort_values('Risk', ascending=False).iloc[0]['Project_ID']}** "
            "has the highest risk based on cash-flow volatility.",

        "4️⃣ Which project creates the highest value (NPV)?":
            f"Project **{df.sort_values('NPV', ascending=False).iloc[0]['Project_ID']}** "
            "has the highest Net Present Value, indicating strong long-term value creation.",

        "5️⃣ What is the overall capital allocation recommendation?":
            "The system recommends prioritizing projects that maximize "
            "risk-adjusted value within the available budget, while "
            "deferring lower-ranked projects due to capital constraints."
    }

    return answers
