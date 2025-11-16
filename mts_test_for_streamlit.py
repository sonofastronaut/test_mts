import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

st.title("Предсказание оттока клиентов (Churn) — ML Модель")

# --- File upload ---
uploaded = st.file_uploader("Загрузите CSV-файл с данными", type="csv")

if uploaded is None:
    st.info("Пожалуйста, загрузите файл, чтобы продолжить")
    st.stop()

# --- Load data ---
df = pd.read_csv(uploaded)
st.subheader("Первые строки данных:")
st.write(df.head())

# --- Basic info ---
st.subheader("Информация о данных")
st.write("Количество пропусков по столбцам:")
st.write(df.isna().sum())

# --- Preprocessing ---
if "customerid" in df.columns:
    df.drop("customerid", axis=1, inplace=True)

# Encoding categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# --- Train model button ---
if st.button("Обучить модель"):
    if "churn" not in df.columns:
        st.error("Ошибка: В данных нет столбца 'churn'")
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

    st.success("Модель обучена!")

    st.subheader("Метрики модели")
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.write("ROC-AUC:", roc_auc_score(y_test, probs))

    st.text("Classification Report:")
    st.text(classification_report(y_test, preds))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, preds))
