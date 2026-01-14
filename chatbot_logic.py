# ---------------- DECISION EXPLAINER ----------------
if page == "4️⃣ Explainer Chatbot":
    st.header("Capital Allocation Explainer")
    st.markdown(
        """
        This section provides predefined executive-level questions 
        with clear, consistent explanations of the capital allocation decision.
        """
    )

    # Ensure allocation data exists
    if "df" not in locals():
        st.warning("Please run the Capital Allocation section first.")
    else:
        questions = {
            "1️⃣ Which projects were selected for funding?":
                lambda df: (
                    "The following projects were selected for funding based on "
                    "their strong risk-adjusted returns and capital efficiency:\n\n"
                    + ", ".join(
                        df[df["Decision"].str.contains("Selected")]["Project_ID"]
                    )
                ),

            "2️⃣ Which projects were rejected due to budget constraints?":
                lambda df: (
                    "The following projects were not funded due to limited capital "
                    "availability, despite having potential long-term value:\n\n"
                    + ", ".join(
                        df[df["Decision"].str.contains("Rejected")]["Project_ID"]
                    )
                ),

            "3️⃣ Which project carries the highest risk?":
                lambda df: (
                    f"Project **{df.sort_values('Risk', ascending=False).iloc[0]['Project_ID']}** "
                    "has the highest risk, measured by cash-flow volatility relative to "
                    "expected returns."
                ),

            "4️⃣ Which project creates the highest value (NPV)?":
                lambda df: (
                    f"Project **{df.sort_values('NPV', ascending=False).iloc[0]['Project_ID']}** "
                    "generates the highest Net Present Value, indicating strong long-term "
                    "value creation."
                ),

            "5️⃣ What is the overall capital allocation recommendation?":
                lambda df: (
                    "The system recommends prioritizing projects that maximize "
                    "risk-adjusted value within the available capital budget. "
                    "High-return, lower-risk projects are funded first, while "
                    "lower-ranked projects are deferred due to capital constraints."
                )
        }

        selected_question = st.radio(
            "Select a question to view the explanation:",
            list(questions.keys())
        )

        st.info(questions[selected_question](df))
