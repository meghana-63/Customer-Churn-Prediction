import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("Data/churn.csv")

# Features and target
X = df[["Age", "MonthlyCharges", "Tenure"]]
y = df["Churn"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Title
st.title("Customer Churn Prediction")

st.write("Predict whether a customer is likely to leave or stay.")

# ===== ADD GRAPH HERE =====

st.subheader("Customer Churn Distribution")

fig, ax = plt.subplots()

df["Churn"].value_counts().plot(
    kind="bar",
    ax=ax
)

ax.set_title("Customer Churn Count")
ax.set_xlabel("Churn")
ax.set_ylabel("Customers")

st.pyplot(fig)

# ===== INPUT BOXES START HERE =====

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

charges = st.number_input(
    "Monthly Charges",
    min_value=100,
    max_value=5000,
    value=1000
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=1,
    max_value=60,
    value=12
)

if st.button("Predict"):

    prediction = model.predict([[age, charges, tenure]])

    if prediction[0] == 1:
        st.error("Customer is likely to leave.")
    else:
        st.success("Customer is likely to stay.")