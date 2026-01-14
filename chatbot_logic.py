def get_predefined_answers(df):
    selected = df[df["Decision"].str.contains("Selected")]["Project_ID"].tolist()
    rejected = df[df["Decision"].str.contains("Rejected")]["Project_ID"].tolist()

    highest_risk = df.sort_values("Risk", ascending=False).iloc[0]
    highest_npv = df.sort_values("NPV", ascending=False).iloc[0]

    return {
        "1️⃣ Which projects were selected for funding?":
            (
                "These projects were selected because they offer a strong balance of "
                "expected returns, manageable risk, and efficient use of capital. "
                "They provide better value compared to other options and fit well within "
                "the available investment budget.\n\n"
                + (", ".join(selected) if selected else "No projects were selected.")
            ),

        "2️⃣ Which projects were rejected due to budget constraints?":
            (
                "These projects were not funded because, when compared with other options, "
                "they delivered lower value for the level of risk involved. With limited "
                "capital available, priority was given to projects that offer higher returns, "
                "faster recovery of investment, and more stable cash flows.\n\n"
                + (", ".join(rejected) if rejected else "No projects were rejected.")
            ),

        "3️⃣ Which project carries the highest risk?":
            (
                f"{highest_risk['Project_ID']} carries the highest risk because its expected "
                "cash flows are more uncertain and show higher volatility compared to other "
                "projects."
            ),

        "4️⃣ Which project creates the highest value (NPV)?":
            (
                f"{highest_npv['Project_ID']} creates the highest value as it is expected to "
                "generate the greatest net benefit over its lifetime, even after accounting "
                "for the cost of capital."
            ),

        "5️⃣ What is the overall capital allocation recommendation?":
            (
                "The overall recommendation is to invest in projects that deliver higher "
                "returns with lower risk while staying within the available capital limit. "
                "Projects that do not meet this balance can be reconsidered in the future "
                "when more capital becomes available."
            )
    }

