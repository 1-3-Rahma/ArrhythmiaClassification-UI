import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# File paths
model_path = "random_forest_model.pkl"
encoder_path = "label_encoder.pkl"

# Load model and encoder
model, label_encoder = None, None
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error(f"❌ Model file not found: {model_path}")

if os.path.exists(encoder_path):
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
else:
    st.error(f"❌ Encoder file not found: {encoder_path}")

# Required feature names
features = ['0_pre-RR', '0_post-RR', '0_pPeak', '0_tPeak', '0_rPeak', '0_sPeak', '0_qPeak',
            '0_qrs_interval', '0_pq_interval', '0_qt_interval', '0_st_interval',
            '0_qrs_morph0', '0_qrs_morph1', '0_qrs_morph2', '0_qrs_morph3', '0_qrs_morph4',
            '1_pre-RR', '1_post-RR', '1_pPeak', '1_tPeak', '1_rPeak', '1_sPeak', '1_qPeak',
            '1_qrs_interval', '1_pq_interval', '1_qt_interval', '1_st_interval',
            '1_qrs_morph0', '1_qrs_morph1', '1_qrs_morph2', '1_qrs_morph3', '1_qrs_morph4']

# Advice dictionary
advice = {
    "N": "✅ Normal beat. No immediate action required.",
    "Q": "❓ Unknown type. Consult your cardiologist.",
    "SVEB": "⚠️ Supraventricular ectopic beat. Suggest ECG monitoring.",
    "VEB": "🚨 Ventricular ectopic beat. High risk, consult a doctor.",
    "F": "🧪 Fusion beat detected. Further tests may be needed."
}

# Prediction function
def predict(df):
    encoded_preds = model.predict(df)
    return label_encoder.inverse_transform(encoded_preds)

# Streamlit App UI
st.title("🫀 Arrhythmia Classification App")
st.markdown("This app classifies ECG signals into different types of arrhythmia using a trained **Random Forest** model.")

if model is not None and label_encoder is not None:

    st.sidebar.header("📌 Choose Input Method")
    option = st.sidebar.radio("How would you like to input data?", ("Upload CSV", "Enter Manually"))

    if option == "Upload CSV":
       uploaded_file = st.file_uploader("Upload your ECG readings CSV", type=["csv"])
       if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                try:
                    data = pd.read_csv(uploaded_file, encoding="latin1")
                except Exception as e:
                    st.error(f"🚫 Error reading file: {e}")
                    data = None

            if data is not None:
                if all(f in data.columns for f in features):
                    predictions = predict(data)
                    data['Predicted Arrhythmia'] = predictions
                    st.success("Predictions completed!")
                    st.write(data)
                    st.info(advice.get(predictions[0], "ℹ️ No specific advice available."))    
                else:
                    st.error("Your CSV must contain the required features.")
            else:
                st.error("🚫 Error reading file.")


    elif option == "Enter Manually":
        st.markdown("📝 Enter ECG values manually")
        with st.form("manual_entry_form"):
            user_input = {feature: st.number_input(f"{feature}", value=0.0) for feature in features}
            submitted = st.form_submit_button("Submit")

        if submitted:
            input_df = pd.DataFrame([user_input])
            prediction = predict(input_df)
            st.success(f"🎯 Predicted Arrhythmia: **{prediction[0]}**")
            st.info(advice.get(prediction[0], "ℹ️ No specific advice available."))

    
    
    st.markdown("---")
    st.subheader("ℹ️ About This Model")
    st.write("""
    This Random Forest model was trained on ECG features to classify arrhythmia into:
    - **N**: Normal
    - **Q**: Unknown
    - **SVEB**: Supraventricular ectopic beat
    - **VEB**: Ventricular ectopic beat
    - **F**: Fusion beat

    The model was trained and evaluated using appropriate metrics. For clinical use, always consult with a healthcare provider.
    """)
