import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("💼 Employee Attrition Prediction System")

# -----------------------------
# LOAD MODELS
# -----------------------------
model = joblib.load('saved_models/best_model.pkl')
scaler = joblib.load('saved_models/scaler.pkl')
feature_cols = joblib.load('saved_models/feature_columns.pkl')
kmeans = joblib.load('saved_models/kmeans.pkl')
pca = joblib.load('saved_models/pca.pkl')

# -----------------------------
# SIDEBAR INPUT (CLEAN UI)
# -----------------------------
st.sidebar.header("🧾 Employee Details")

def user_input():
    age = st.sidebar.slider("Age", 18, 60, 30)
    income = st.sidebar.number_input("Monthly Income", 1000, 50000, 5000)
    years = st.sidebar.slider("Years at Company", 0, 40, 5)

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
    marital = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    dept = st.sidebar.selectbox("Department", ["Sales", "HR", "R&D"])
    role = st.sidebar.selectbox("Job Role", [
        "Manager","Sales Executive","Research Scientist",
        "Laboratory Technician","HR","Sales Representative"
    ])

    job_sat = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
    work_life = st.sidebar.slider("Work-Life Balance", 1, 4, 3)
    performance = st.sidebar.slider("Performance Rating", 1, 5, 3)

    # Convert to dataframe
    data = pd.DataFrame({
        'Age':[age],
        'Monthly_Income':[income],
        'Years_at_Company':[years],
        'Gender':[1 if gender=="Male" else 0],
        'Overtime':[1 if overtime=="Yes" else 0],
        'Marital_Status':[marital],
        'Job_Satisfaction':[job_sat],
        'Work_Life_Balance':[work_life],
        'Performance_Rating':[performance],
        'Department':[dept],
        'Job_Role':[role]
    })

    return data

input_df = user_input()

# -----------------------------
# PREPROCESS INPUT (MATCH TRAINING)
# -----------------------------
def preprocess_input(df):
    
    # One-hot encoding
    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=feature_cols, fill_value=0)

    return df

processed = preprocess_input(input_df)
scaled = scaler.transform(processed)

# -----------------------------
# PREDICTION
# -----------------------------
st.subheader("📊 Prediction")

prob = model.predict_proba(scaled)[0][1]
pred = model.predict(scaled)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("Attrition Probability", f"{prob:.2f}")

with col2:
    if pred == 1:
        st.error("⚠️ High Risk Employee")
    else:
        st.success("✅ Likely to Stay")

# -----------------------------
# CLUSTERING
# -----------------------------
cluster = kmeans.predict(scaled)[0]
st.info(f"📌 Employee belongs to Cluster: {cluster}")

# -----------------------------
# PCA VISUALIZATION
# -----------------------------
st.subheader("📉 Employee Position (PCA)")

user_pca = pca.transform(scaled)

fig, ax = plt.subplots()
ax.scatter(user_pca[:,0], user_pca[:,1])
ax.set_title("Employee in PCA Space")

st.pyplot(fig)

# -----------------------------
# SHAP EXPLAINABILITY
# -----------------------------
st.subheader("🔍 Why this prediction? (SHAP)")

explainer = shap.Explainer(model)
shap_values = explainer(processed)

fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0, :, 1], show=False)
st.pyplot(fig)

