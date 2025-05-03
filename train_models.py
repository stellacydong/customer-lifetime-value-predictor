import pandas as pd
import numpy as np
import joblib
import os

from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load model-ready dataset
df = pd.read_csv("data/clv_churn_dataset.csv")

# Features and targets
features = ['Recency_x', 'Frequency', 'Monetary']
X = df[features]
y_clv = df['CLV']
y_churn = df['Churned']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_clv_train, y_clv_test = train_test_split(X_scaled, y_clv, test_size=0.2, random_state=42)
_, _, y_churn_train, y_churn_test = train_test_split(X_scaled, y_churn, test_size=0.2, random_state=42)

# Train CLV model
xgb_clf = XGBRegressor(n_estimators=100, random_state=42)
xgb_clf.fit(X_train, y_clv_train)

# Train Churn model
xgb_churn = XGBClassifier(n_estimators=100, random_state=42)
xgb_churn.fit(X_train, y_churn_train)

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_clf, "models/xgb_clv_model.pkl")
joblib.dump(xgb_churn, "models/xgb_churn_model.pkl")
joblib.dump(scaler, "models/rfm_scaler.pkl")

print("✅ Models and scaler saved in ./models/")
