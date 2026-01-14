import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data_generation import *
from forecasting import *
from financial_metrics import *
from allocation_model import *
from chatbot_logic import get_predefined_answers

st.set_page_config(layout="wide")

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigate",
    ["1️⃣ Company Overview",
     "2️⃣ AI Forecasting",
     "3️⃣ Capital Allocation",
     "4️⃣ Explainer Chatbot"]
)

scenario = st.sidebar.selectbox("Scenario", ["Base", "Best", "Worst"])
wacc = st.sidebar.slider("Cost of Capital (WACC)", 0.09, 0.13, 0.11, 0.01)

# ---------------- DATA ----------------
historical = generate_historical_data()
projects = generate_project_data()

# ---------------- PAGE 1 ----------------
if page == "1️⃣ Company Overview":
    st.title("AI-Driven Capital Allocation Advisor")

    st.markdown("""
**Company:** Apex Industries Ltd. *(Fictional)*  
**Profile:** Diversified Indian Conglomerate  
**Decision Maker:** CFO & Capital Allocation Committee  

All values in ₹ Crore (INR Cr).
""")

    st.subheader("Historical Financial Data")
    st.dataframe(historical)

    st.subheader("Investment Projects")
    st.dataframe(projects)

# ---------------- PAGE 2 ----------------
if page == "2️⃣ AI Forecasting":
    st.header("AI Forecasting")

    target = st.selectbox("Forecast Variable", ["Revenue", "Operating_Cost"])
    best_name, _, results = train_and_select_model(historical, target)

    df_results = pd.DataFrame({
        "Model": results.keys(),
        "R2": [results[m]["R2"] for m in results],
        "MAE": [results[m]["MAE"] for m in results]
    })

    st.dataframe(df_results)
    st.success(f"Selected Model: {best_name}")

# ---------------- PAGE 3 ----------------
if page == "3️⃣ Capital Allocation":
    st.header(f"Capital Allocation – {scenario} Scenario")

    _, rev_model, _ = train_and_select_model(historical, "Revenue")
    _, cost_model, _ = train_and_select_model(historical, "Operating_Cost")

    latest = historical[["Year", "Inflation (%)", "Demand_Index"]].iloc[[-1]]
    forecast_revenue = rev_model.predict(latest)[0]
    forecast_cost = cost_model.predict(latest)[0]

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

    # ✅ STORE RESULT
    st.session_state["allocation_df"] = df

    st.dataframe(df)
    st.success(f"Capital Used: ₹{spent} Cr | Capital Unused: ₹{100 - spent} Cr")

    fig, ax = plt.subplots()
    ax.scatter(df["Risk"], df["NPV"])
    ax.set_xlabel("Risk")
    ax.set_ylabel("NPV")
    ax.set_title("Risk vs Return")
    st.pyplot(fig)

# ---------------- PAGE 4 ----------------
if page == "4️⃣ Explainer Chatbot":
    st.header("Capital Allocation Explainer")

    if "allocation_df" not in st.session_state:
        st.warning("Please run Capital Allocation first.")
    else:
        df = st.session_state["allocation_df"]
        answers = get_predefined_answers(df)

        q = st.radio("Select a question", list(answers.keys()))
        st.info(answers[q])
