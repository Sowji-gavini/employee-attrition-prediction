import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("💼 Employee Attrition Prediction & Segmentation")

# -----------------------------
# TRAIN MODEL (CACHED)
# -----------------------------
@st.cache_resource
def train_model():

    df = pd.read_csv("/Users/gavinisowjanya/Downloads/employee_attrition_dataset.csv")

    df.dropna(inplace=True)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
    df['Overtime'] = le.fit_transform(df['Overtime'])

    df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})

    df = pd.get_dummies(df, columns=['Department','Job_Role'], drop_first=True)

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for model in models.values():
        model.fit(X_train, y_train)

    # Unsupervised
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return models, scaler, X.columns, kmeans, pca, X_pca, clusters


models, scaler, columns, kmeans, pca, X_pca, clusters = train_model()

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Enter Employee Details")

def user_input():
    data = {}
    for col in columns:
        data[col] = st.sidebar.number_input(col, value=0.0)
    return pd.DataFrame([data])

input_df = user_input()

# Align columns
input_df = input_df.reindex(columns=columns, fill_value=0)

scaled = scaler.transform(input_df)

# -----------------------------
# PREDICTIONS
# -----------------------------
st.subheader("📊 Prediction Probabilities")

pred_results = {}
for name, model in models.items():
    prob = model.predict_proba(scaled)[0][1]
    pred_results[name] = prob

st.bar_chart(pred_results)

# Final prediction (Random Forest)
final_pred = models["Random Forest"].predict(scaled)[0]

if final_pred == 1:
    st.error("⚠️ High Risk of Attrition")
else:
    st.success("✅ Likely to Stay")

# -----------------------------
# CLUSTER
# -----------------------------
cluster = kmeans.predict(scaled)[0]
st.info(f"Cluster Group: {cluster}")

# -----------------------------
# PCA VISUALIZATION
# -----------------------------
st.subheader("📉 PCA Visualization")

fig, ax = plt.subplots()
ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters, alpha=0.5)

user_point = pca.transform(scaled)
ax.scatter(user_point[:,0], user_point[:,1], c='red', s=100)

ax.set_title("Employee Segmentation")

st.pyplot(fig)
