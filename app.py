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
    [" Company Overview",
     " Forecasting",
     " Capital Allocation",
     " Explainer Chatbot"]
)

scenario = st.sidebar.selectbox("Scenario", ["Base", "Best", "Worst"])
wacc = st.sidebar.slider("Cost of Capital (WACC)", 0.09, 0.13, 0.11, 0.01)

# ---------------- DATA ----------------
historical = generate_historical_data()
projects = generate_project_data()

# ---------------- PAGE 1 ----------------
if page == " Company Overview":
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
if page == " Forecasting":
    st.markdown("##  AI Forecasting")
    st.markdown(
        """
        This section uses machine learning models to forecast key financial variables 
        using historical company data and economic indicators.
        """
    )

    st.markdown("---")

    # Select variable to forecast
    target = st.selectbox(
        " Forecast Variable",
        ["Revenue", "Operating_Cost"]
    )

    best_name, best_model, results = train_and_select_model(historical, target)

    # ---------------- MODEL COMPARISON ----------------
    st.subheader(" Model Performance Comparison")

    df_results = pd.DataFrame({
        "Model": results.keys(),
        "R² Score": [round(results[m]["R2"], 3) for m in results],
        "Mean Absolute Error (₹ Cr)": [round(results[m]["MAE"], 2) for m in results]
    })

    st.dataframe(df_results)

    st.markdown("---")

    # ---------------- METRIC EXPLANATION ----------------
    st.subheader(" How to read these metrics")
    st.markdown(
        """
        - **R² Score** shows how well the model explains past trends.  
          A value closer to **1** means the model fits historical data very well.  

        - **Mean Absolute Error (MAE)** shows the average difference between actual 
          and predicted values in ₹ Crore.  
          Lower MAE means more accurate forecasts.
        """
    )

    # ---------------- FORECAST VS ACTUAL ----------------
    st.markdown("---")
    st.subheader(" Forecast vs Actual Comparison")

    X = historical[["Year", "Inflation (%)", "Demand_Index"]]
    y_actual = historical[target]
    y_pred = best_model.predict(X)

    fig, ax = plt.subplots()
    ax.plot(historical["Year"], y_actual, label="Actual", marker="o")
    ax.plot(historical["Year"], y_pred, label="Forecast", marker="o", linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{target} (₹ Cr)")
    ax.set_title(f"Actual vs Forecasted {target}")
    ax.legend()

    st.pyplot(fig)

    st.markdown(
        """
        **Interpretation:**  
        The closer the forecasted line is to the actual values, the more reliable 
        the model is in capturing historical trends.
        """
    )

    # ---------------- MODEL SELECTION ----------------
    st.markdown("---")
    st.subheader(" Model Selection Decision")

    st.success(f"Selected Model: **{best_name}**")

    if best_name == "Linear Regression":
        st.info(
            """
            Linear Regression was selected because it demonstrates **very high accuracy**
            (high R² score) and **low prediction error** (low MAE).  
            It also provides **stable and interpretable forecasts**, which is important 
            for financial decision-making.
            """
        )

    # ---------------- WHY NOT DECISION TREE ----------------
    st.markdown("---")
    st.subheader(" Why Decision Tree was not selected")

    st.markdown(
        """
        Although the Decision Tree model can capture complex patterns, it performed 
        poorly in this case due to limited historical data.

        - It showed **lower accuracy (lower R² score)**  
        - It produced **higher prediction error (higher MAE)**  
        - It is more prone to **overfitting**, making forecasts less stable  

        For financial forecasting, **stability and consistency** are preferred over 
        complexity.
        """
    )

    # ---------------- CONFIDENCE EXPLANATION ----------------
    st.markdown("---")
    st.subheader(" Confidence in Forecasts")

    st.markdown(
        """
        Forecast confidence is based on:
        - Strong historical fit (high R²)
        - Low prediction error (low MAE)
        - Consistent trend capture over time  

        While no forecast is perfect, the selected model provides a **reliable baseline**
        for evaluating future project performance and supporting capital allocation decisions.
        """
    )

    # ---------------- BUSINESS TAKEAWAY ----------------
    st.markdown("---")
    st.subheader(" Business Takeaway")

    st.markdown(
        """
        The AI forecasting model provides dependable estimates of future financial values.
        These forecasts are used as inputs for project evaluation, risk assessment, 
        and capital allocation decisions in later stages of the system.
        """
    )


# ---------------- PAGE 3: CAPITAL ALLOCATION ----------------
if page == " Capital Allocation":

    st.header(f"Capital Allocation – {scenario} Scenario")

    st.markdown(
        """
        This section evaluates competing investment projects under different economic
        scenarios and allocates capital based on expected returns, risk, and budget constraints.
        """
    )

    # --------- FORECAST INPUTS ----------
    _, rev_model, _ = train_and_select_model(historical, "Revenue")
    _, cost_model, _ = train_and_select_model(historical, "Operating_Cost")

    latest_inputs = historical[["Year", "Inflation (%)", "Demand_Index"]].iloc[[-1]]
    forecast_revenue = rev_model.predict(latest_inputs)[0]
    forecast_cost = cost_model.predict(latest_inputs)[0]

    # --------- PROJECT EVALUATION ----------
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

    # --------- SCORING & ALLOCATION ----------
    df = score_projects(df)
    df, spent = allocate(df)

    # Save for chatbot
    st.session_state["allocation_df"] = df

    # --------- RESULTS TABLE ----------
    st.subheader(" Project Evaluation & Allocation Results")
    st.dataframe(df)

    st.success(f"Capital Used: ₹{spent} Cr | Capital Unused: ₹{100 - spent} Cr")

    # --------- SCENARIO EXPLANATION ----------
    st.markdown("---")
    st.subheader(" Scenario-Based Allocation Explanation")

    if scenario == "Best":
        st.info(
            "This is the **Best Case scenario**, assuming favourable market conditions and strong growth. "
            "The allocation strategy focuses on maximising value creation, even if it involves taking "
            "slightly higher risk."
        )
    elif scenario == "Base":
        st.info(
            "This is the **Base Case scenario**, representing normal business conditions. "
            "Projects are selected based on a balanced trade-off between risk and return, "
            "with emphasis on stability and efficient capital use."
        )
    else:
        st.info(
            "This is the **Worst Case scenario**, reflecting adverse economic conditions. "
            "The allocation becomes conservative, prioritising financially resilient projects "
            "that limit downside risk."
        )

    # --------- RISK–RETURN EXPLANATION ----------
    st.markdown("---")
    st.subheader(f" Risk vs Return Analysis – {scenario} Scenario")

    st.markdown(
        """
        This chart compares projects based on:
        - **Risk:** Uncertainty in project cash flows  
        - **Return:** Net Present Value (NPV) in ₹ Crore
        """
    )

    if scenario == "Best":
        st.markdown(
            "In favourable conditions, projects with higher risk may still be acceptable "
            "if they offer significantly higher returns."
        )
    elif scenario == "Base":
        st.markdown(
            "Under normal conditions, preference is given to projects that deliver "
            "consistent returns without excessive risk."
        )
    else:
        st.markdown(
            "During adverse conditions, priority is given to projects in the **upper-left region** "
            "(lower risk, acceptable return) to protect capital."
        )

    # --------- IMPROVED RISK VS RETURN GRAPH ----------
    fig, ax = plt.subplots(figsize=(9, 6))

    # Color code selected vs rejected
    colors = df["Decision"].apply(
        lambda x: "green" if "Selected" in x else "red"
    )

    ax.scatter(df["Risk"], df["NPV"], c=colors, s=120, alpha=0.8)

    # Annotate points
    for _, row in df.iterrows():
        ax.annotate(
            row["Project_ID"],
            (row["Risk"], row["NPV"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10
        )

    # Reference lines
    ax.axhline(df["NPV"].median(), linestyle="--", alpha=0.4)
    ax.axvline(df["Risk"].median(), linestyle="--", alpha=0.4)

    ax.set_xlabel("Risk (Cash-Flow Volatility)")
    ax.set_ylabel("Return (NPV in ₹ Crore)")
    ax.set_title(f"Project Risk vs Return – {scenario} Scenario")

    st.pyplot(fig)

    st.info(
        """
        **How to read this chart:**  
        - 🟢 Green points = Funded projects  
        - 🔴 Red points = Rejected projects  
        - Upper-left area = Better return with lower risk  
        - Lower-right area = Higher risk with weaker returns
        """
    )

    # --------- FINAL DECISION SUMMARY ----------
    st.markdown("---")
    st.subheader(" Capital Allocation Decision Summary")

    st.markdown(
        f"""
        Under the **{scenario} Scenario**, capital allocation decisions are aligned with the
        company’s risk appetite and expected market conditions. Projects were funded or
        deferred based on how effectively they balance value creation, risk exposure,
        and efficient use of limited capital.
        """
    )




# ---------------- PAGE 4 ----------------
if page == " Explainer Chatbot":
    st.markdown("##  Capital Allocation Explainer")
    st.markdown(
        """
        <div style="color:#6b7280; font-size:15px;">
        Executive-level insights explaining how and why capital allocation decisions were made.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if "allocation_df" not in st.session_state:
        st.warning("⚠️ Please run the Capital Allocation step first.")
    else:
        df = st.session_state["allocation_df"]
        answers = get_predefined_answers(df)

        # Layout: Left = questions | Right = answer
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("###  Key Questions")
            selected_q = st.radio(
                "",
                list(answers.keys()),
                label_visibility="collapsed"
            )

        with col2:
            st.markdown("###  Explanation")
            st.markdown(
                f"""
                <div style="
                    background-color:#f8fafc;
                    padding:20px;
                    border-radius:10px;
                    border-left:5px solid #2563eb;
                    font-size:16px;
                    line-height:1.6;
                ">
                {answers[selected_q]}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown(
            """
            <div style="font-size:13px; color:#6b7280;">
            📌 Note: This system provides decision support based on financial analysis. 
            Final investment decisions remain with management.
            </div>
            """,
            unsafe_allow_html=True
        )

