
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("Bank Marketing Prediction")
st.write("Predict whether a customer will subscribe to a term deposit.")

# Load preprocessing pipeline
preprocessor = joblib.load("models/preprocessor.pkl")

# Available trained models
model_files = {
    "Logistic Regression": "models/Logistic_Regression.pkl",
    "Decision Tree": "models/Decision_Tree.pkl",
    "KNN": "models/KNN.pkl",
    "Naive Bayes": "models/Naive_Bayes.pkl",
    "SVM": "models/SVM.pkl"
}

st.subheader("Select Machine Learning Model")

selected_model = st.selectbox(
    "Choose a model",
    list(model_files.keys()),
    index=4
)

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 35)

    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur",
        "housemaid", "management", "retired",
        "self-employed", "services", "student",
        "technician", "unemployed", "unknown"
    ])

    marital = st.selectbox("Marital Status", [
        "married", "single", "divorced", "unknown"
    ])

    education = st.selectbox("Education", [
        "primary", "secondary", "tertiary", "unknown"
    ])

with col2:
    default = st.selectbox("Credit in Default", ["no", "yes"])

    balance = st.number_input("Account Balance", value=1000)

    housing = st.selectbox("Housing Loan", ["no", "yes"])

    loan = st.selectbox("Personal Loan", ["no", "yes"])

with col3:
    contact = st.selectbox("Contact Type", [
        "cellular", "telephone", "unknown"
    ])

    day = st.number_input("Last Contact Day", 1, 31, 15)

    month = st.selectbox("Last Contact Month", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])

    campaign = st.number_input(
        "Campaign Contacts", 1, 100, 1
    )

st.subheader("Previous Campaign Information")

col4, col5 = st.columns(2)

with col4:
    pdays = st.number_input(
        "Days Since Previous Contact",
        -1, 999, -1
    )

with col5:
    previous = st.number_input(
        "Previous Contacts",
        0, 500, 0
    )

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    ["unknown", "failure", "other", "success"]
)

# Prediction
if st.button("Predict Subscription", type="primary"):

    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    # Preprocess customer information
    processed_data = preprocessor.transform(input_data)

    # Load selected model
    model = joblib.load(model_files[selected_model])

    # Convert input for Naive Bayes
    if selected_model == "Naive Bayes":
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()

    # Generate prediction
    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(
        processed_data
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Customer is predicted to SUBSCRIBE!")
    else:
        st.warning("Customer is predicted NOT TO SUBSCRIBE.")

    st.metric(
        "Subscription Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(f"Selected Model: {selected_model}")
