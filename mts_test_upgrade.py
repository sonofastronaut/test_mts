import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score,
    f1_score, precision_recall_curve, auc
)

# Настройки страницы
st.set_page_config(page_title="Churn Prediction App", page_icon="📊")
st.title("📉 Предсказание оттока клиентов (Churn)")

# ============================================================
# 1. Загрузка данных
# ============================================================
st.header("📁 Загрузите CSV-файл")
uploaded = st.file_uploader("Выберите файл", type=["csv"])

if uploaded is None:
    st.info("Пожалуйста, загрузите CSV-файл, чтобы продолжить")
    st.stop()

df = pd.read_csv(uploaded)

# ============================================================
# 2. Предварительный анализ
# ============================================================
st.header("🔎 Предпросмотр данных")
st.write(df.head())

st.subheader("📊 Пропуски в данных")
missing = df.isna().sum().to_frame(name="Количество пропусков")
st.write(missing)

# Распределение целевой переменной
if "churn" in df.columns:
    st.subheader("📌 Распределение целевой переменной — churn")

    fig, ax = plt.subplots()
    sns.countplot(x=df["churn"], ax=ax)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Количество")
    st.pyplot(fig)
else:
    st.warning("⚠️ В данных нет столбца 'churn'. Обучение модели будет недоступно.")

# ============================================================
# 3. Предобработка
# ============================================================

# Удаляем ID, если есть
if "customerid" in df.columns:
    df.drop("customerid", axis=1, inplace=True)

# Кодирование категорий
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col])

# ============================================================
# 4. Корреляции
# ============================================================
st.subheader("🧩 Матрица корреляций")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="Reds")
st.pyplot(fig)

# ============================================================
# 5. Обучение модели
# ============================================================
st.header("🤖 Обучение модели")

if st.button("Обучить модель"):
    
    if "churn" not in df.columns:
        st.error("❌ Ошибка: нет столбца 'churn'. Невозможно обучить модель.")
        st.stop()

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Модель
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    st.success("🎉 Модель успешно обучена!")

    # ============================================================
    # 6. Метрики (для несбалансированных данных)
    # ============================================================
    st.subheader("📈 Метрики модели")

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, probs)

    # PR-AUC (главная метрика при дисбалансе)
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall_curve, precision_curve)

    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}  ← ключевая метрика для churn")
    st.write(f"**F1-score:** {f1:.4f}")
    st.write(f"**ROC-AUC:** {roc_auc:.4f}")
    st.write(f"**PR-AUC:** {pr_auc:.4f}")

    # st.text("Классификационный отчёт:")
    # st.text(classification_report(y_test, preds))

    # ============================================================
    # 7. Precision–Recall Curve
    # ============================================================
    st.subheader("📉 Precision–Recall Curve")

    fig, ax = plt.subplots()
    ax.plot(recall_curve, precision_curve)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    st.pyplot(fig)

    # ============================================================
    # 8. Матрица ошибок
    # ============================================================
    st.subheader("🟥 Матрица ошибок")

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Фактически")
    st.pyplot(fig)

    # ============================================================
    # 9. Важность признаков
    # ============================================================
    st.subheader("🌲 Важность признаков")

    importances = pd.Series(model.feature_importances_, index=X.columns)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

