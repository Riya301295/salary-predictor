# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(page_title="Income Class Predictor", layout="wide")
st.title("💼 Income Classification App (<=50K or >50K)")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

df = load_data()

# Encode target
le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])  # >50K = 1, <=50K = 0

# One-hot encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
dtree = DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid = GridSearchCV(dtree, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# User input
st.header("🧍 Enter User Data")
user_data = {}
for col in df.columns:
    if col != 'income':
        if df[col].dtype == object:
            user_data[col] = st.selectbox(col, df[col].unique())
        else:
            user_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))

user_df = pd.DataFrame([user_data])
user_encoded = pd.get_dummies(user_df)
missing_cols = set(X.columns) - set(user_encoded.columns)
for col in missing_cols:
    user_encoded[col] = 0
user_encoded = user_encoded[X.columns]  # reorder columns

# Predict
if st.button("🔮 Predict Income Class"):
    pred = best_model.predict(user_encoded)[0]
    label = ">50K" if pred == 1 else "<=50K"
    st.subheader(f"🎯 Predicted Income: **{label}**")

    # Accuracy
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"✅ Model Accuracy: **{acc*100:.2f}%**")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # ROC Curve
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    st.subheader("🚦 ROC Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

    # Feature Importance
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(15)
    st.subheader("📌 Top Feature Importances")
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax3)
    st.pyplot(fig3)

