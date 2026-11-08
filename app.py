
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("Bank Marketing Prediction")
st.write("Predict whether a customer will subscribe to a term deposit.")

@st.cache_resource
def load_preprocessor():
    return joblib.load("models/preprocessor.pkl")


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


preprocessor = load_preprocessor()

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

    processed_data = preprocessor.transform(input_data)

    model = load_model(model_files[selected_model])

    if selected_model == "Naive Bayes":
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()

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

    if prediction == 1 and probability < 0.5:
        st.info(
            "Note: The SVM classification decision and its "
            "probability estimate can differ."
        )

st.divider()

st.title("Model Evaluation")

st.write(
    "Upload the test dataset to evaluate the selected "
    "machine learning model."
)

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_df = pd.read_csv(uploaded_file)

        if "y" not in test_df.columns:
            st.error("The uploaded CSV must contain the target column 'y'.")
        else:
            st.write("Uploaded Dataset Shape:", test_df.shape)

            X_test = test_df.drop(
                columns=["y", "duration"],
                errors="ignore"
            )

            y_test = test_df["y"]

            if y_test.dtype == "object":
                y_test = y_test.map({"no": 0, "yes": 1})

            if y_test.isna().any():
                st.error("Invalid target values found in column 'y'.")
            else:
                expected_columns = list(
                    preprocessor.feature_names_in_
                )

                missing_columns = [
                    col for col in expected_columns
                    if col not in X_test.columns
                ]

                if missing_columns:
                    st.error(
                        f"Missing columns: {missing_columns}"
                    )
                else:
                    X_test = X_test[expected_columns]

                    X_processed = preprocessor.transform(X_test)

                    model = load_model(model_files[selected_model])

                    if selected_model == "Naive Bayes":
                        if hasattr(X_processed, "toarray"):
                            X_processed = X_processed.toarray()

                    predictions = model.predict(X_processed)

                    probabilities = model.predict_proba(
                        X_processed
                    )[:, 1]

                    st.success("Evaluation completed successfully!")

                    st.subheader("Performance Metrics")

                    accuracy = accuracy_score(
                        y_test, predictions
                    )

                    precision = precision_score(
                        y_test, predictions,
                        zero_division=0
                    )

                    recall = recall_score(
                        y_test, predictions,
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_test, predictions,
                        zero_division=0
                    )

                    auc = roc_auc_score(
                        y_test, probabilities
                    )

                    mcc = matthews_corrcoef(
                        y_test, predictions
                    )

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    metric_col1.metric(
                        "Accuracy", f"{accuracy:.4f}"
                    )

                    metric_col2.metric(
                        "Precision", f"{precision:.4f}"
                    )

                    metric_col3.metric(
                        "Recall", f"{recall:.4f}"
                    )

                    metric_col4, metric_col5, metric_col6 = st.columns(3)

                    metric_col4.metric(
                        "F1 Score", f"{f1:.4f}"
                    )

                    metric_col5.metric(
                        "AUC", f"{auc:.4f}"
                    )

                    metric_col6.metric(
                        "MCC", f"{mcc:.4f}"
                    )

                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y_test,
                        predictions,
                        labels=[0, 1]
                    )

                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual No", "Actual Yes"],
                        columns=["Predicted No", "Predicted Yes"]
                    )

                    st.dataframe(cm_df)

                    st.subheader("Classification Report")

                    report = classification_report(
                        y_test,
                        predictions,
                        target_names=[
                            "No Subscription",
                            "Subscription"
                        ],
                        output_dict=True,
                        zero_division=0
                    )

                    st.dataframe(
                        pd.DataFrame(report).transpose()
                    )

                    st.caption(
                        f"Evaluation Model: {selected_model}"
                    )

    except Exception as e:
        st.error(f"Evaluation error: {e}")
