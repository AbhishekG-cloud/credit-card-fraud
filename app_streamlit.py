import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredPipeline
from demo_transactions import LEGIT, FRAUD

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
### About This Project

This application predicts whether a credit card transaction is fraudulent using an XGBoost classifier trained on the Kaggle Credit Card Fraud Detection Dataset.

**Dataset Size:** 284,807 Transactions  
**Fraud Rate:** 0.17%  
**Model:** XGBoost Classifier
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Load Legitimate Example"):
        st.session_state.transaction = LEGIT

with col2:
    if st.button("🔴 Load Fraud Example"):
        st.session_state.transaction = FRAUD

if "transaction" in st.session_state:

    txn = st.session_state.transaction.copy()

    st.subheader("Selected Transaction")

    display_data = txn.copy()
    display_data.pop("Class", None)

    st.dataframe(display_data)

    if st.button("🚀 Predict"):

        data = CustomData(
            Time=txn["Time"],
            Amount=txn["Amount"],
            V1=txn["V1"],
            V2=txn["V2"],
            V3=txn["V3"],
            V4=txn["V4"],
            V5=txn["V5"],
            V6=txn["V6"],
            V7=txn["V7"],
            V8=txn["V8"],
            V9=txn["V9"],
            V10=txn["V10"],
            V11=txn["V11"],
            V12=txn["V12"],
            V13=txn["V13"],
            V14=txn["V14"],
            V15=txn["V15"],
            V16=txn["V16"],
            V17=txn["V17"],
            V18=txn["V18"],
            V19=txn["V19"],
            V20=txn["V20"],
            V21=txn["V21"],
            V22=txn["V22"],
            V23=txn["V23"],
            V24=txn["V24"],
            V25=txn["V25"],
            V26=txn["V26"],
            V27=txn["V27"],
            V28=txn["V28"]
        )

        pred_df = data.get_data_as_dataframe()

        pipeline = PredPipeline()

        prediction, probability = pipeline.predict(pred_df)

        fraud_prob = float(probability[0][1] * 100)

        st.divider()

        if prediction[0] == 1:

            st.error(
                f"⚠️ Fraudulent Transaction Detected\n\nConfidence: {fraud_prob:.2f}%"
            )

        else:

            st.success(
                f"✅ Legitimate Transaction\n\nConfidence: {100 - fraud_prob:.2f}%"
            )

else:
    st.info(
        "Select either 'Load Legitimate Example' or 'Load Fraud Example' to test the model."
    )