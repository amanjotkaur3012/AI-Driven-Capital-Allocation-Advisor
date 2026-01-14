import streamlit as st
import pandas as pd

from data_generation import *
from forecasting import *
from financial_metrics import *
from allocation_model import *

st.set_page_config(layout="wide")
st.title("AI-Driven Capital Allocation Advisor")

historical = generate_historical_data()
projects = generate_project_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Page", [
    "Overview",
    "Forecasting & Model Comparison",
    "Capital Allocation"
])

if page == "Overview":
    st.subheader("Project Overview")
    st.write("AI-assisted capital allocation using interpretable ML models.")
    st.dataframe(projects)

if page == "Forecasting & Model Comparison":
    st.subheader("ML Model Comparison")

    target = st.selectbox("Forecast Target", ["Revenue", "Operating_Cost"])
    results = train_models(historical, target)

    model_choice = st.radio(
        "Choose Model for Forecasting",
        list(results.keys())
    )

    st.write("### Model Performance")
    st.metric("R² Score", round(results[model_choice]["r2"], 3))
    st.metric("MAE", round(results[model_choice]["mae"], 2))

    st.info(
        "Linear Regression offers higher transparency, while Decision Tree "
        "captures non-linear patterns with controlled complexity."
    )

if page == "Capital Allocation":
    st.subheader("Capital Allocation Results")

    records = []
    for _, p in projects.iterrows():
        cashflows = estimate_cashflows(50, 30, p["Project_Life"])

        records.append({
            "Project_ID": p["Project_ID"],
            "Initial_Investment": p["Initial_Investment"],
            "NPV": npv(cashflows, p["Initial_Investment"]),
            "IRR": irr(cashflows, p["Initial_Investment"]),
            "Payback": payback(cashflows, p["Initial_Investment"]),
            "Risk": risk(cashflows)
        })

    df = pd.DataFrame(records)
    df = score_projects(df)
    df = allocate_budget(df)

    st.dataframe(df)
