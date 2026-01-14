def get_chatbot_response(question, df):
    """
    Rule-based explainer chatbot.
    Reads final allocation output and explains decisions.
    """
    q = question.lower()

    if "rejected" in q:
        rejected = df[df["Decision"].str.contains("Rejected")]
        return f"Projects rejected due to budget constraint: {', '.join(rejected['Project_ID'])}"

    if "selected" in q:
        selected = df[df["Decision"].str.contains("Selected")]
        return f"Funded projects: {', '.join(selected['Project_ID'])}"

    if "risky" in q:
        risky = df.sort_values("Risk", ascending=False).iloc[0]
        return f"{risky['Project_ID']} is the riskiest project based on cash flow volatility."

    if "npv" in q:
        top = df.sort_values("NPV", ascending=False).iloc[0]
        return f"{top['Project_ID']} has the highest NPV."

    return (
        "I can explain project selection, rejection, risk, and return. "
        "Please ask a capital allocation related question."
    )
