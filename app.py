
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Insights App", layout="wide")

st.title("📊 Smart Customer Insights Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload customer RFM data (.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Preview Uploaded Data")
    st.write(df.head())

    # Load models and scaler
    xgb_clf = joblib.load("models/xgb_clv_model.pkl")
    xgb_churn = joblib.load("models/xgb_churn_model.pkl")
    scaler = joblib.load("models/rfm_scaler.pkl")

    # Prepare data
    features = ['Recency_x', 'Frequency', 'Monetary']
    X = scaler.transform(df[features])

    # Predict CLV and Churn
    df['Predicted_CLV'] = xgb_clf.predict(X)
    df['Predicted_Churn'] = xgb_churn.predict(X)

    st.subheader("📈 Predictions")
    st.write(df[['CustomerID', 'Predicted_CLV', 'Predicted_Churn']])

    # SHAP Explainability
    explainer = shap.Explainer(xgb_clf)
    shap_values = explainer(X)

    st.subheader("🔍 SHAP Feature Importance (Sample)")
    shap.plots.beeswarm(shap_values, max_display=5, show=False)
    st.pyplot(bbox_inches="tight")

    # Optional: Auto-suggestions
    def suggest_action(row):
        if row['Predicted_Churn'] == 1:
            return "⚠️ Offer retention discount"
        elif row['Predicted_CLV'] > df['Predicted_CLV'].quantile(0.75):
            return "🌟 VIP Customer – Priority Support"
        else:
            return "👍 Maintain Engagement"

    df['Suggested_Action'] = df.apply(suggest_action, axis=1)
    st.subheader("🎯 Suggested Actions")
    st.dataframe(df[['CustomerID', 'Predicted_CLV', 'Predicted_Churn', 'Suggested_Action']])
else:
    st.info("Please upload a customer RFM dataset (.csv) to begin.")
