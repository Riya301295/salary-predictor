import streamlit as st
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor

st.title("Salary Prediction App")
st.write("Upload a dataset to train the model and predict salaries based on years of experience.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", data.head())

    if 'YearsExperience' in data.columns and 'Salary' in data.columns:
        X = data[['YearsExperience']]
        y = data['Salary']

        # Train model
        model = DecisionTreeRegressor(random_state=0)
        model.fit(X, y)

        # Save model for prediction
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        st.success("Model trained successfully!")

        # Input for prediction
        exp = st.number_input("Enter Years of Experience", min_value=0.0, step=0.1)
        if st.button("Predict Salary"):
            salary = model.predict([[exp]])[0]
            st.success(f"Predicted Salary: ₹ {salary:,.2f}")
    else:
        st.error("CSV must contain 'YearsExperience' and 'Salary' columns.")