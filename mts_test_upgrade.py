import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

st.set_page_config(page_title="Churn Prediction App", page_icon="📊")

st.title("📉 Предсказание оттока клиентов (Churn)")

# --- File upload ---
st.header("📁 Загрузите файл")
uploaded = st.file_uploader("CSV-файл", type="csv")

if uploaded is None:
    st.info("Пожалуйста, загрузите CSV-файл")
    st.stop()

# --- Load data ---
df = pd.read_csv(uploaded)
st.header("🔎 Предпросмотр данных")
st.write(df.head())

# --- Basic info ---
st.header("📊 Информация о данных")
st.subheader("Пропуски")
st.write(df.isna().sum())

# --- Graph #1: Target distribution ---
if "churn" in df.columns:
    st.subheader("📌 Распределение целевой переменной — churn")

    fig, ax = plt.subplots()
    sns.countplot(x=df["churn"], ax=ax)
    ax.set_title("Распределение churn")
    st.pyplot(fig)

# --- Preprocessing ---
if "customerid" in df.columns:
    df.drop("customerid", axis=1, inplace=True)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# --- Graph #2: Correlation heatmap ---
st.subheader("🧩 Корреляционная матрица")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap="Blues")
st.pyplot(fig)

# --- Train model button ---
st.header("🤖 Обучение модели")
if st.button("Обучить модель"):

    if "churn" not in df.columns:
        st.error("В данных нет столбца 'churn'")
        st.stop()

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    st.success("Модель успешно обучена!")

    st.subheader("📈 Метрики модели")
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.write("ROC-AUC:", roc_auc_score(y_test, probs))

    st.text("Classification report:")
    st.text(classification_report(y_test, preds))

    # --- Graph #3: Confusion Matrix ---
    st.subheader("🟥 Confusion Matrix")

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
    st.pyplot(fig)

    # --- Graph #4: Feature importance ---
    st.subheader("🌲 Важность признаков (Feature Importance)")

    importances = pd.Series(model.feature_importances_, index=X.columns)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)
