import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Salary Prediction App")
st.write("Upload a dataset to train the model and predict salaries based on years of experience.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")

        # Check required columns
        if 'YearsExperience' in data.columns and 'Salary' in data.columns:
            st.write("### Dataset Preview:")
            st.dataframe(data.head())

            # Scatter plot
            fig1, ax1 = plt.subplots()
            sns.scatterplot(x='YearsExperience', y='Salary', data=data, ax=ax1)
            ax1.set_title("Years of Experience vs Salary")
            st.pyplot(fig1)

            # Split data
            X = data[['YearsExperience']]
            y = data['Salary']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Accuracy metrics
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("📊 Model Performance")
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")

            # Actual vs predicted plot
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_test, y_pred, color='green')
            ax2.set_xlabel("Actual Salary")
            ax2.set_ylabel("Predicted Salary")
            ax2.set_title("Actual vs Predicted Salary")
            st.pyplot(fig2)

            # Regression line on test set
            fig3, ax3 = plt.subplots()
            sns.scatterplot(x=X_test['YearsExperience'], y=y_test, label='Actual', ax=ax3)
            sns.lineplot(x=X_test['YearsExperience'], y=y_pred, color='red', label='Predicted', ax=ax3)
            ax3.set_title("Regression Line vs Actual (Test Set)")
            st.pyplot(fig3)

            # Prediction Input
            st.write("### Predict Salary")
            experience = st.number_input("Enter Years of Experience", min_value=0.0, step=0.1)

            if st.button("Predict"):
                prediction = model.predict(np.array([[experience]]))
                st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")

        else:
            st.error("CSV must contain 'YearsExperience' and 'Salary' columns.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a dataset CSV file.")

  
   
  
   
