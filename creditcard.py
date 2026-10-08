import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "Machine Learning application for detecting potentially fraudulent "
    "credit card transactions."
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)


uploaded_file = st.file_uploader(
    "Upload Credit Card Transaction Dataset",
    type=["csv"]
)


if uploaded_file is None:
    st.info("Please upload a CSV dataset to start.")

    st.markdown("""
    ### Required Dataset Format

    Your CSV file should contain transaction features and a target column
    named `Class`.

    Example:

    | Time | V1 | V2 | V3 | Amount | Class |
    |------|----|----|----|--------|-------|
    | 100  | 1.2 | -0.4 | 0.8 | 50.00 | 0 |
    | 200  | -2.1 | 1.3 | -1.2 | 900.00 | 1 |

    Where:

    - `0` → Legitimate transaction
    - `1` → Fraudulent transaction
    """)

    st.stop()


# --------------------------------------------------
# Read Dataset
# --------------------------------------------------

try:
    data = load_data(uploaded_file)

except Exception as e:
    st.error(f"Unable to read dataset: {e}")
    st.stop()


# --------------------------------------------------
# Validate Dataset
# --------------------------------------------------

if "Class" not in data.columns:
    st.error("Dataset must contain a 'Class' column.")
    st.stop()


st.subheader("📊 Dataset Preview")
st.dataframe(data.head())


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", len(data))

with col2:
    fraud_count = int((data["Class"] == 1).sum())
    st.metric("Fraud Transactions", fraud_count)

with col3:
    legitimate_count = int((data["Class"] == 0).sum())
    st.metric("Legitimate Transactions", legitimate_count)


# --------------------------------------------------
# Prepare Data
# --------------------------------------------------

data = data.dropna()

X = data.drop("Class", axis=1)
y = data["Class"]


# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

if X.shape[1] == 0:
    st.error("Dataset does not contain numeric features.")
    st.stop()


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Train Machine Learning Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("🤖 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Test Transactions",
        len(X_test)
    )


st.write("### Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

st.write("### Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Legitimate", "Actual Fraud"],
    columns=["Predicted Legitimate", "Predicted Fraud"]
)

st.dataframe(cm_df)


# --------------------------------------------------
# Fraud Prediction
# --------------------------------------------------

st.subheader("🔍 Check a Transaction")

st.write(
    "Enter values for the transaction features below."
)

input_values = []

for feature in X.columns:

    value = st.number_input(
        f"{feature}",
        value=0.0
    )

    input_values.append(value)


if st.button("🚨 Detect Fraud"):

    transaction = np.array(input_values).reshape(1, -1)

    transaction_scaled = scaler.transform(transaction)

    prediction = model.predict(transaction_scaled)[0]

    probability = model.predict_proba(transaction_scaled)[0]

    fraud_probability = probability[1] * 100


    if prediction == 1:

        st.error(
            f"⚠️ Potential Fraud Detected!\n\n"
            f"Fraud Probability: {fraud_probability:.2f}%"
        )

    else:

        st.success(
            f"✅ Transaction appears legitimate.\n\n"
            f"Fraud Probability: {fraud_probability:.2f}%"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Credit Card Fraud Detection System | "
    "Machine Learning Project"
)
