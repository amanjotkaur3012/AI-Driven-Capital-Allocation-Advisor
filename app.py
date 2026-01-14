import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_generation import *
from forecasting import *
from financial_metrics import *
from allocation_model import *
from chatbot_logic import get_chatbot_response

st.set_page_config(layout="wide")

# ---------------- TITLE ----------------
st.title("AI-Driven Capital Allocation Advisor")

st.markdown("""
### Company Context

**Company:** Apex Industries Ltd. *(Fictional)*  
**Profile:** Diversified Indian Conglomerate  
**Decision Maker:** CFO & Capital Allocation Committee  

This AI system supports strategic capital allocation decisions.
Final decisions remain with management.

**All monetary values are in ₹ Crore (INR Cr).**
""")

# ---------------- DATA ----------------
historical = generate_historical_data()
projects = generate_project_data()

# ---------------- SIDEBAR ----------------
scenario = st.sidebar.selectbox("Scenario", ["Base", "Best", "Worst"])
wacc = st.sidebar.slider("Cost of Capital (WACC)", 0.09, 0.13, 0.11, 0.01)

page = st.sidebar.radio(
    "Navigate",
    [" Company Overview",
     " AI Forecasting",
     " Capital Allocation",
     " Explainer Chatbot"]
)

# ---------------- OVERVIEW ----------------
if page == " Company Overview":
    st.header("Company Data Overview")

    st.subheader("Historical Financial Data (2018–2024)")
    st.dataframe(historical)

    st.subheader("Internal Investment Projects")
    st.dataframe(projects)

# ---------------- FORECASTING ----------------
if page == " AI Forecasting":
    st.header("AI Forecasting & Model Comparison")

    target = st.selectbox("Forecast Variable", ["Revenue", "Operating_Cost"])
    best_name, _, results = train_and_select_model(historical, target)

    comparison = pd.DataFrame({
        "Model": results.keys(),
        "R² Score": [results[m]["R2"] for m in results],
        "MAE (₹ Cr)": [results[m]["MAE"] for m in results]
    })

    st.dataframe(comparison)
    st.success(f"Selected Model: **{best_name}**")

# ---------------- CAPITAL ALLOCATION ----------------
if page == " Capital Allocation":
    st.header(f"Capital Allocation – {scenario} Scenario")

    _, rev_model, _ = train_and_select_model(historical, "Revenue")
    _, cost_model, _ = train_and_select_model(historical, "Operating_Cost")

    latest_inputs = historical[["Year", "Inflation (%)", "Demand_Index"]].iloc[[-1]]
    forecast_revenue = rev_model.predict(latest_inputs)[0]
    forecast_cost = cost_model.predict(latest_inputs)[0]

    records = []
    for _, p in projects.iterrows():
        cf = cashflows(
            forecast_revenue,
            forecast_cost,
            p["Project_Life (Years)"],
            scenario
        )

        records.append({
            "Project_ID": p["Project_ID"],
            "Investment": p["Initial_Investment (₹ Cr)"],
            "NPV": npv(cf, p["Initial_Investment (₹ Cr)"], wacc),
            "IRR": irr(cf, p["Initial_Investment (₹ Cr)"]),
            "Payback": payback(cf, p["Initial_Investment (₹ Cr)"]),
            "Risk": risk(cf)
        })

    df = pd.DataFrame(records)
    df = score_projects(df)
    df, spent = allocate(df)

    st.dataframe(df)
    st.success(f"Capital Used: ₹{spent} Cr | Capital Unused: ₹{100 - spent} Cr")

    fig, ax = plt.subplots()
    ax.scatter(df["Risk"], df["NPV"])
    ax.set_xlabel("Risk")
    ax.set_ylabel("NPV (₹ Cr)")
    ax.set_title("Risk–Return Trade-off")
    st.pyplot(fig)

# ---------------- CHATBOT ----------------
if page == " Explainer Chatbot":
    st.header("Capital Allocation Explainer")

    st.markdown(
        """
        This section provides predefined executive-level explanations
        of the capital allocation decision.
        """
    )

    if "df" not in locals():
        st.warning("Please complete the Capital Allocation step first.")
    else:
        from chatbot_logic import get_predefined_answers

        answers = get_predefined_answers(df)

        selected_question = st.radio(
            "Select a question to view the explanation:",
            list(answers.keys())
        )

        st.info(answers[selected_question])
