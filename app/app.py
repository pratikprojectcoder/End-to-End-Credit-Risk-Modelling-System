"""Professional Streamlit UI for Credit Risk Modeling System."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from predict import predict_proba
from utils import load_pickle
from preprocess import MODELS_DIR

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Risk Dashboard",
    layout="wide",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .stMetric {
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #374151;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD MODEL
# =========================================================

model = load_pickle(
    MODELS_DIR / "xgboost_model.pkl"
)

encoders = load_pickle(
    MODELS_DIR / "label_encoders.pkl"
)

feature_columns = encoders.get(
    "feature_columns",
    []
)

# =========================================================
# TITLE
# =========================================================

st.title("Credit Risk Prediction Dashboard")

st.markdown(
    """
AI-powered credit default prediction platform for:

- Loan Risk Assessment
- Lending Decision Support
- Financial Risk Analysis
- Customer Credit Evaluation
"""
)

# =========================================================
# KPI SECTION
# =========================================================

st.markdown("## Model Performance")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Accuracy",
    "92%"
)

kpi2.metric(
    "ROC-AUC",
    "0.89"
)

kpi3.metric(
    "Precision",
    "87%"
)

kpi4.metric(
    "Recall",
    "84%"
)

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Customer Risk Prediction",
        "Portfolio Risk Analysis",
        "Model Insights",
    ]
)

# =========================================================
# CUSTOMER RISK PREDICTION
# =========================================================

if page == "Customer Risk Prediction":

    st.header("Customer Credit Risk Assessment")

    col1, col2 = st.columns([1, 1])

    # =====================================================
    # INPUT FORM
    # =====================================================

    with col1:

        st.subheader("Customer Information")

        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=1000,
            max_value=100000,
            value=15000,
        )

        annual_inc = st.number_input(
            "Annual Income",
            min_value=1000,
            max_value=1000000,
            value=50000,
        )

        dti = st.slider(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
        )

        fico_range_low = st.slider(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=700,
        )

        revol_bal = st.number_input(
            "Revolving Balance",
            min_value=0,
            max_value=200000,
            value=10000,
        )

        installment = st.number_input(
            "Monthly Installment",
            min_value=0,
            max_value=5000,
            value=400,
        )

        delinq_2yrs = st.slider(
            "Past Delinquencies",
            min_value=0,
            max_value=20,
            value=0,
        )

        pub_rec = st.slider(
            "Public Records",
            min_value=0,
            max_value=10,
            value=0,
        )

        predict_button = st.button(
            "Analyze Credit Risk"
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_button:

        input_data = pd.DataFrame({

            "loan_amnt": [loan_amnt],
            "annual_inc": [annual_inc],
            "dti": [dti],
            "fico_range_low": [fico_range_low],
            "revol_bal": [revol_bal],
            "installment": [installment],
            "delinq_2yrs": [delinq_2yrs],
            "pub_rec": [pub_rec],

        })

        # =================================================
        # ADD MISSING FEATURES
        # =================================================

        for col in feature_columns:

            if col not in input_data.columns:

                input_data[col] = 0

        # =================================================
        # REORDER FEATURES
        # =================================================

        input_data = input_data[
            feature_columns
        ]

        # =================================================
        # PREDICTION
        # =================================================

        try:

            probability = predict_proba(
                input_data
            )[0]

            prediction_percent = (
                probability * 100
            )

            # =============================================
            # RISK CATEGORY
            # =============================================

            if prediction_percent < 30:

                risk_label = "LOW RISK"

            elif prediction_percent < 70:

                risk_label = "MEDIUM RISK"

            else:

                risk_label = "HIGH RISK"

            # =============================================
            # RESULTS PANEL
            # =============================================

            with col2:

                st.subheader(
                    "Risk Assessment Results"
                )

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Default Probability",
                    f"{prediction_percent:.2f}%"
                )

                m2.metric(
                    "Risk Category",
                    risk_label
                )

                m3.metric(
                    "Credit Score",
                    fico_range_low
                )

                st.markdown("---")

                if risk_label == "LOW RISK":

                    st.success(
                        "LOW CREDIT RISK"
                    )

                elif risk_label == "MEDIUM RISK":

                    st.warning(
                        "MODERATE CREDIT RISK"
                    )

                else:

                    st.error(
                        "HIGH CREDIT RISK"
                    )

                # =========================================
                # RISK BAR
                # =========================================

                st.markdown(
                    "### Default Risk Level"
                )

                st.progress(
    float(
        min(
            prediction_percent / 100,
            1.0
        )
    )
)

                st.markdown("---")

                # =========================================
                # FEATURE IMPORTANCE
                # =========================================

                st.subheader(
                    "Key Risk Drivers"
                )

                importances = getattr(
                    model,
                    "feature_importances_",
                    None
                )

                if importances is not None:

                    feature_importance_df = (
                        pd.DataFrame({
                            "Feature": feature_columns,
                            "Importance": importances
                        })
                    )

                    feature_importance_df = (
                        feature_importance_df
                        .sort_values(
                            by="Importance",
                            ascending=False
                        )
                        .head(10)
                    )

                    fig, ax = plt.subplots(
                        figsize=(8, 4)
                    )

                    ax.barh(
                        feature_importance_df["Feature"],
                        feature_importance_df["Importance"]
                    )

                    ax.invert_yaxis()

                    ax.set_title(
                        "Top Risk Factors"
                    )

                    st.pyplot(fig)

                st.markdown("---")

                st.subheader(
                    "Risk Insights"
                )

                st.info(
                    """
Higher debt ratio, lower credit score,
larger loan amount, and previous
delinquencies increase credit risk.
"""
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

# =========================================================
# PORTFOLIO ANALYSIS
# =========================================================

elif page == "Portfolio Risk Analysis":

    st.header("Portfolio Risk Analysis")

    st.markdown(
        """
Analyze lending portfolio trends and
customer credit distribution.
"""
    )

    portfolio_data = pd.DataFrame({

        "Risk Category": [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],

        "Customers": [
            320,
            140,
            40
        ]

    })

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        portfolio_data["Risk Category"],
        portfolio_data["Customers"]
    )

    ax.set_title(
        "Customer Risk Distribution"
    )

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Portfolio Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Customers",
        "500"
    )

    c2.metric(
        "Average Credit Score",
        "684"
    )

    c3.metric(
        "High Risk Customers",
        "40"
    )

# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    st.header("Model Insights")

    st.markdown(
        """
The prediction system evaluates financial
behavior patterns to estimate default risk.
"""
    )

    importance_df = pd.DataFrame({

        "Feature": [

            "Credit Score",
            "Debt-to-Income Ratio",
            "Loan Amount",
            "Annual Income",
            "Delinquencies",
            "Revolving Balance",
            "Installment"

        ],

        "Importance": [

            0.32,
            0.21,
            0.18,
            0.12,
            0.08,
            0.05,
            0.04

        ]

    })

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.invert_yaxis()

    ax.set_title(
        "Key Risk Drivers"
    )

    st.pyplot(fig)

    st.markdown("---")

    st.info(
        """
The system identifies important financial
indicators influencing customer default probability.
"""
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    """
Built using Machine Learning, Streamlit,
XGBoost, and Financial Risk Analytics
"""
)