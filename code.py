# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Set page config
st.set_page_config(page_title="Income Prediction", layout="wide")

st.title("📊 Predict Income Category (<=50K or >50K)")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Strip column names and string values
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop rows with missing values (optional)
    df.dropna(inplace=True)

    # Encode target column 'income'
    if 'income' in df.columns:
        le = LabelEncoder()
        df['income'] = le.fit_transform(df['income'])
    else:
        st.error("Dataset must contain the 'income' column")
        st.stop()

    # One-hot encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Split features and target
    X = df.drop('income', axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train Decision Tree with GridSearch
    model = DecisionTreeClassifier(random_state=42)
    params = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Accuracy: {acc*100:.2f}%")

    # Show classification report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig_cm)

    # ROC Curve
    st.subheader("ROC Curve")
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    fig_roc = plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    st.pyplot(fig_roc)

    # Feature Importance
    st.subheader("Top 15 Important Features")
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(15)

    fig_feat = plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features, y=top_features.index)
    plt.title('Feature Importances')
    st.pyplot(fig_feat)

else:
    st.info("📂 Please upload a valid dataset CSV with an 'income' column to proceed.")