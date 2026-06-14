import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc

# load model and scaler
model = load_model("credit_card_fraud_model.keras")
scaler = joblib.load("scaler.pkl")

# ================= SIDEBAR =================
st.sidebar.title("💳 Fraud Detection System")

page = st.sidebar.selectbox("Navigation", [
    "Home",
    "Dataset Info",
    "Live Prediction",
    "Model Insights"
])

# ================= HOME PAGE =================
if page == "Home":
    st.title("💳 Credit Card Fraud Detection")
    st.write("AI-powered system to detect fraudulent transactions using ANN model.")

#DATASET INFO
elif page == "Dataset Info":
    st.title("📊 Dataset Information")
    st.write("Credit Card Fraud dataset contains 30 features:")
    st.write("- V1 to V28 (PCA transformed features)")
    st.write("- Time")
    st.write("- Amount")
    st.write("Target: Class (0 = Legit, 1 = Fraud)")


# ================= LIVE PREDICTION =================
elif page == "Live Prediction":

    st.title("🔮 Live Fraud Detection")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing transactions",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.write(df.head())

        # store true labels if available
        y_true = None
        if 'Class' in df.columns:
            y_true = df['Class'].copy()
            df = df.drop('Class', axis=1)

        # expected features
        expected_features = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

        # check missing columns
        missing = [col for col in expected_features if col not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
            st.stop()

        # enforce correct order
        df = df[expected_features]

        st.write("Before scaling shape:", df.shape)

        # scale
        df_scaled = scaler.transform(df)

        st.write("After scaling shape:", df_scaled.shape)

        # predict
        predictions = model.predict(df_scaled)

        # convert back for display
        result_df = pd.DataFrame(df, columns=expected_features)
        result_df["Fraud_Probability"] = predictions
        result_df["Prediction"] = np.where(
            result_df["Fraud_Probability"] > 0.5,
            "Fraud",
            "Legitimate"
        )

        # store for insights page
        st.session_state["results"] = result_df
        st.session_state["y_true"] = y_true

        fraud_count = (result_df["Prediction"] == "Fraud").sum()

        st.subheader("Results")
        st.metric("Fraud Transactions Detected", fraud_count)
        st.metric("Total Transactions", len(result_df))

        st.write(result_df.head())

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )
        
elif page == "Model Insights":
    st.title("📊 Model Evaluation (Real Metrics)")

    if "results" not in st.session_state:
        st.warning("Run prediction first.")
        st.stop()

    df = st.session_state["results"]
    y_true = st.session_state.get("y_true")

    st.subheader("Prediction Summary")

    fraud_count = (df["Prediction"] == "Fraud").sum()
    legit_count = (df["Prediction"] == "Legit").sum()

    st.metric("Total Transactions", len(df))
    st.metric("Fraud Detected", fraud_count)
    st.metric("Legit Transactions", legit_count)

    import matplotlib.pyplot as plt

    # ================= CONFUSION MATRIX =================
    if y_true is not None:
        st.subheader("Confusion Matrix")

        y_pred = (df["Fraud_Probability"] > 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        ax.imshow(cm)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # ================= ROC CURVE =================
    st.subheader("ROC Curve")

    if y_true is not None:
        fpr, tpr, _ = roc_curve(y_true, df["Fraud_Probability"])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        st.pyplot(fig)
    else:
        st.info("ROC curve needs dataset with 'Class' column")