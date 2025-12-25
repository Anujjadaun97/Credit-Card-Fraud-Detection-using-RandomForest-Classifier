import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import traceback

st.title("🛡️ Credit Card Fraud Detection System")
st.write("This application uses Machine Learning to detect fraudulent credit card transactions in real-time.")

# --- Model loading with clear error messages ---
MODEL_PATH = Path("models/fraud_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

if not MODEL_PATH.exists():
    st.error(f"Model file not found at {MODEL_PATH}. Create 'models' and save the trained model as 'fraud_model.pkl'.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    st.info("✅ Model loaded successfully")
except Exception as e:
    st.error("Failed to load model. See details below:")
    st.text(traceback.format_exc())
    st.stop()

# Optional: load scaler if you used one to preprocess inputs
scaler = None
if SCALER_PATH.exists():
    try:
        scaler = joblib.load(SCALER_PATH)
        st.info("Scaler loaded (optional)")
    except Exception:
        st.warning("Failed to load scaler; continuing without it.")

# Helper to find the CSV (allow either 'data/creditcard.csv' or 'creditcard.csv')
def find_creditcard_csv():
    for p in (Path("data/creditcard.csv"), Path("creditcard.csv")):
        if p.exists():
            return p
    return None


# Sidebar for user input method
input_option = st.sidebar.radio("Select Input Method", ["Upload CSV", "Manual Input", "Random Simulation"])

if input_option == "Random Simulation":
    st.header("Simulate a Transaction")
    if st.button("Generate Random Transaction"):
        csv_path = find_creditcard_csv()
        if csv_path is None:
            st.error("Could not find 'creditcard.csv' (look for 'creditcard.csv' or 'data/creditcard.csv').")
        else:
            try:
                df = pd.read_csv(csv_path)
                if 'Class' not in df.columns:
                    st.error("The CSV file does not contain a 'Class' column.")
                else:
                    df_sample = df.sample(min(100, len(df)))
                    random_transaction = df_sample.sample(1)
                    true_label = random_transaction['Class'].values[0]
                    features = random_transaction.drop('Class', axis=1)

                    st.write("### Transaction Details (Features V1-V28, Time, Amount)")
                    st.dataframe(features)

                    X = features.values
                    expected = getattr(model, 'n_features_in_', None)
                    if expected is not None and X.shape[1] != expected:
                        st.error(f"Model expects {expected} features but the data has {X.shape[1]}. Check feature columns/order.")
                    else:
                        try:
                            pred = model.predict(X)
                            proba = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None

                            st.write("---")
                            st.write("### Prediction Result")

                            if pred[0] == 1:
                                if proba is not None:
                                    st.error(f"🚨 FRAUD DETECTED! (Confidence: {proba:.2%})")
                                else:
                                    st.error("🚨 FRAUD DETECTED!")
                            else:
                                if proba is not None:
                                    st.success(f"✅ Legitimate Transaction (Confidence: { (1-proba):.2%})")
                                else:
                                    st.success("✅ Legitimate Transaction")

                            st.write(f"**Actual Label:** {'Fraud' if true_label == 1 else 'Legit'}")
                        except Exception:
                            st.error("Prediction failed — see details below:")
                            st.text(traceback.format_exc())
            except Exception:
                st.error("Failed to load or sample CSV:")
                st.text(traceback.format_exc())

elif input_option == "Manual Input":
    st.write("Enter the transaction details (V1-V28 are PCA components).")
    # Simple text input for features (simplified for demo)
    input_str = st.text_area("Paste comma separated values for V1...V28, Amount, Time (30 values)", 
                             "0.5, -0.2, ...")
    if st.button("Predict"):
        try:
            input_list = [float(x.strip()) for x in input_str.split(',') if x.strip() != '']
            features = np.array(input_list).reshape(1, -1)
            expected = getattr(model, 'n_features_in_', None)
            if expected is not None and features.shape[1] != expected:
                st.warning(f"Model expects {expected} features but you provided {features.shape[1]}. Please provide the correct number of values.")
            else:
                # apply scaler if available and appropriate
                if scaler is not None:
                    try:
                        # only transform if shapes match
                        if np.array(scaler.mean_).size == features.shape[1]:
                            features = scaler.transform(features)
                        else:
                            st.info("Loaded scaler shape doesn't match provided input; skipping scaling.")
                    except Exception:
                        st.warning("Scaler transform failed; continuing without scaling.")

                try:
                    prediction = model.predict(features)
                    if prediction[0] == 1:
                        st.error("Fraudulent Transaction")
                    else:
                        st.success("Legitimate Transaction")
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)[0][1]
                        st.write(f"Confidence of Fraud: {proba:.2%}")
                except Exception:
                    st.error("Prediction failed — see details below:")
                    st.text(traceback.format_exc())
        except Exception:
            st.warning("Please enter valid comma-separated numerical values.")