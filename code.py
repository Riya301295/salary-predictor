import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Title
st.title("💼 Salary Classification Predictor")
st.write("This app predicts if a person earns more than ₹50K/year based on their details.")
st.markdown("---")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

df = load_data()

# Show dataset preview
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)

# Encode categorical features
def preprocess_data(df):
    df = df.copy()
    le_dict = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

df_encoded, label_encoders = preprocess_data(df)

# Split data
X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show metrics
accuracy = accuracy_score(y_test, y_pred)
st.subheader("🔍 Model Accuracy")
st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

# Confusion Matrix
st.subheader("📊 Confusion Matrix")
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
ax.matshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], va='center', ha='center')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)

# Classification Report
st.subheader("📃 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
st.pyplot(plt)

# User Input Section
st.markdown("---")
st.subheader("🧑‍💻 Predict a New Person's Salary")

# Example-based input
exp = st.text_input("Years of Experience (e.g., 5.0)", placeholder="5.0")
education = st.text_input("Education Level (e.g., Bachelors)", placeholder="Bachelors")
job = st.text_input("Job Title (e.g., Engineer)", placeholder="Engineer")
gender = st.text_input("Gender (e.g., Male/Female)", placeholder="Female")
age = st.text_input("Age (e.g., 28)", placeholder="28")

# Validate input
if st.button("🔮 Predict Salary"):
    try:
        new_data = pd.DataFrame({
            "YearsExperience": [float(exp)],
            "Education": [education],
            "JobTitle": [job],
            "Gender": [gender],
            "Age": [int(age)]
        })

        # Encode using trained label encoders
        for col in ["Education", "JobTitle", "Gender"]:
            le = label_encoders.get(col)
            if le:
                if new_data[col][0] in le.classes_:
                    new_data[col] = le.transform(new_data[col])
                else:
                    st.warning(f"'{new_data[col][0]}' not seen in training data. Try a different value.")
                    st.stop()

        # Predict
        prediction = model.predict(new_data)[0]
        pred_proba = model.predict_proba(new_data)[0][prediction]

        st.success(f"Prediction: **{'Salary > ₹50K' if prediction == 1 else 'Salary ≤ ₹50K'}**")
        st.write(f"Prediction Confidence: **{pred_proba * 100:.2f}%**")

    except Exception as e:
        st.error(f"Error: {e}")


