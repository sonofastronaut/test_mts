import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score, precision_score, recall_score,
    f1_score, precision_recall_curve, auc
)

# Настройки страницы
st.set_page_config(page_title="Предсказание оттока абонентов в телекоме", page_icon="📊")
st.title("📉 Предсказание оттока абонентов")

# Загрузка данных
st.header("📁 Загрузите CSV-файл")
uploaded = st.file_uploader("Выберите файл", type=["csv"])

if uploaded is None:
    st.info("Пожалуйста, загрузите CSV-файл, чтобы продолжить")
    st.stop()

df = pd.read_csv(uploaded)

# Предпросмотр данных
st.header("🔎 Предпросмотр данных")
st.write(df.head())

# Пропуски
st.subheader("📊 Пропуски в данных")
missing = df.isna().sum().to_frame(name="Количество пропусков")
st.write(missing)

# Определение целевой переменной
st.subheader("🎯 Определение целевой переменной")

# Ищем бинарные признаки
binary_columns = [col for col in df.columns if df[col].nunique() == 2]

if len(binary_columns) > 0:
    st.info("В датасете обнаружены бинарные столбцы. Выберите целевой признак.")
    target = st.selectbox("Целевая переменная:", binary_columns)
else:
    st.warning("Бинарных признаков не найдено. Выберите целевую переменную вручную.")
    target = st.selectbox("Целевая переменная:", df.columns)

st.write(f"Выбран целевой столбец: **{target}**")

# Распределение целевого признака
st.subheader("📌 Распределение целевой переменной")
fig, ax = plt.subplots()
sns.countplot(x=df[target], ax=ax)
ax.set_xlabel(target)
ax.set_ylabel("Количество")
st.pyplot(fig)

# Предобработка
if "customerid" in df.columns:
    df.drop("customerid", axis=1, inplace=True)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col])

# Корреляции
st.subheader("🧩 Матрица корреляций")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="Reds")
st.pyplot(fig)

# Обучение модели
st.header("🤖 Обучение модели (RandomForest)")

if st.button("Обучить модель"):

    X = df.drop(target, axis=1)
    y = df[target]

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Модель RandomForest 
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Порог классификации
    threshold = 0.3
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    st.success("🎉 Модель успешно обучена!")

    # Метрики
    st.subheader("📈 Метрики модели")

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    roc_auc = roc_auc_score(y_test, probs)

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall_curve, precision_curve)

    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1-score:** {f1:.4f}")
    st.write(f"**ROC-AUC:** {roc_auc:.4f}")
    st.write(f"**PR-AUC:** {pr_auc:.4f}")

    # Кривая Precision–Recall
    fig, ax = plt.subplots()
    ax.plot(recall_curve, precision_curve)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Кривая Precision–Recall")
    st.pyplot(fig)

    # Матрица ошибок
    st.subheader("🟥 Матрица ошибок")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Фактически")
    st.pyplot(fig)

    # Важность признаков
    st.subheader("🌲 Важность признаков")
    importances = pd.Series(model.feature_importances_, index=X.columns)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)


