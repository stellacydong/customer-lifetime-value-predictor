# 💡 Customer Lifetime Value & Churn Predictor

This project analyzes e-commerce customer behavior to:
- Segment customers using RFM and clustering
- Predict Customer Lifetime Value (CLV)
- Classify churn risk
- Provide explainability using SHAP
- Offer strategic suggestions via a Streamlit app

## 📁 Project Structure

```
customer-lifetime-value-predictor/
│
├── data/                  # Contains raw and cleaned data
├── models/                # Saved XGBoost models and scaler
├── app.py                 # Streamlit app script
├── train_models.py        # Script to train and save models
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

## 🚀 Run Locally

### 1. Create virtual environment
```bash
conda create -n clv_env python=3.10
conda activate clv_env
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train models
```bash
python train_models.py
```

### 4. Launch app
```bash
streamlit run app.py
```

## 📦 Features

- Upload customer RFM `.csv`
- Predict CLV and churn
- Visualize feature importance using SHAP
- Auto-suggest business actions

## 📊 Sample Input Format

| CustomerID | Recency_x | Frequency | Monetary |
|------------|-----------|-----------|----------|
| 12345      | 23        | 5         | 200.00   |

## 🧠 Authors
Built by [Your Name] as a demo project for customer analytics and ML explainability.
