# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Step 1: Load Dataset
@st.cache_data

def load_data():
    df = pd.read_csv("dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    le = LabelEncoder()
    df['income'] = le.fit_transform(df['income'])  # >50K = 1, <=50K = 0
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

df = load_data()

# Step 2: Train Model
X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

dtree = DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(dtree, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Step 3: UI Setup
st.title("💰 Income Prediction App")
st.markdown("Enter your information to predict if your income is likely to be >50K or <=50K.")

# Step 4: User Input
def user_input_features():
    age = st.number_input("Age", min_value=18, max_value=90, value=30, help="Example: 30")
    experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, help="Example: 5")
    education = st.selectbox("Education Level", ['Bachelors', 'Masters', 'PhD'], help="Example: Masters")
    job_title = st.text_input("Job Title", "Software Engineer", help="Example: Data Scientist, Manager")
    gender = st.selectbox("Gender", ['Male', 'Female'], help="Select your gender")

    user_df = pd.DataFrame({
        'age': [age],
        'education': [education],
        'job_title': [job_title],
        'gender': [gender],
        'experience': [experience]
    })
    return user_df

input_df = user_input_features()

# Step 5: Preprocess User Input
original = pd.read_csv("dataset.csv")
combined = pd.concat([original, input_df], axis=0)
combined.columns = combined.columns.str.strip()
combined = combined.applymap(lambda x: x.strip() if isinstance(x, str) else x)
categorical_cols = combined.select_dtypes(include='object').columns
combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

input_final = combined.tail(1)
input_final = input_final.reindex(columns=X.columns, fill_value=0)

# Step 6: Predict
prediction = best_model.predict(input_final)[0]
proba = best_model.predict_proba(input_final)[0][1]

st.subheader("🧠 Prediction Result")
st.write("**Predicted Income Class:**", ">50K" if prediction == 1 else "<=50K")
st.write("**Confidence:**", f"{proba*100:.2f}%")

# Step 7: Accuracy & Reports
st.subheader("📈 Model Performance")
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc*100:.2f}%")

# Step 8: Confusion Matrix
st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
plt.title('Confusion Matrix')
st.pyplot(fig1)

# Step 9: ROC Curve
st.subheader("🚦 ROC Curve")
y_prob = best_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], linestyle='--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend(loc="lower right")
st.pyplot(fig2)

# Step 10: Feature Importance
st.subheader("📌 Top 15 Feature Importances")
importances = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_features, y=top_features.index, ax=ax3)
ax3.set_title('Top 15 Features')
ax3.set_xlabel('Importance Score')
st.pyplot(fig3)
