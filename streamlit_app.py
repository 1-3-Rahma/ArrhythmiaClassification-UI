import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model

# File paths
rf_model_path = "random_forest_model.pkl"
encoder_path = "label_encoder.pkl"
cnn_model_path = "ecg_classifier.h5"

# Load model and encoder
rf_model, label_encoder, cnn_model = None, None, None
if os.path.exists(rf_model_path):
    with open(rf_model_path, "rb") as f:
        rf_model = pickle.load(f)
else:
    st.error(f"❌ Random Forest model file not found: {rf_model_path}")

# Load Label Encoder
if os.path.exists(encoder_path):
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
else:
    st.error(f"❌ Encoder file not found: {encoder_path}")

# Load CNN model
if os.path.exists(cnn_model_path):
    try:
        cnn_model = load_model(cnn_model_path)
    except Exception as e:
        st.error(f"❌ Error loading CNN model: {e}")
else:
    st.error(f"❌ CNN model file not found: {cnn_model_path}")

# Required feature names
features = ['0_pre-RR', '0_post-RR', '0_pPeak', '0_tPeak', '0_rPeak', '0_sPeak', '0_qPeak',
            '0_qrs_interval', '0_pq_interval', '0_qt_interval', '0_st_interval',
            '0_qrs_morph0', '0_qrs_morph1', '0_qrs_morph2', '0_qrs_morph3', '0_qrs_morph4',
            '1_pre-RR', '1_post-RR', '1_pPeak', '1_tPeak', '1_rPeak', '1_sPeak', '1_qPeak',
            '1_qrs_interval', '1_pq_interval', '1_qt_interval', '1_st_interval',
            '1_qrs_morph0', '1_qrs_morph1', '1_qrs_morph2', '1_qrs_morph3', '1_qrs_morph4']

# Class mapping for CNN
cnn_class_mapping = {0: 'F', 1: 'N', 2: 'Q', 3: 'SVEB', 4: 'VEB'}

# Advice dictionary
advice = {
    "N": "✅ Normal beat. No immediate action required.",
    "Q": "❓ Unknown type. Consult your cardiologist.",
    "SVEB": "⚠️ Supraventricular ectopic beat. Suggest ECG monitoring.",
    "VEB": "🚨 Ventricular ectopic beat. High risk, consult a doctor.",
    "F": "🧪 Fusion beat detected. Further tests may be needed."
}

# Prediction functions
def predict_with_rf(df):
    encoded_preds = rf_model.predict(df)
    return label_encoder.inverse_transform(encoded_preds)

def predict_with_cnn(img):
     # Ensure image is RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    img = img.resize((128, 128))  # Resize to match model input
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    
    prediction = cnn_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return cnn_class_mapping.get(predicted_class,"Unknown") 

# Streamlit App UI
st.title("🫀 Arrhythmia Classification App")
st.markdown("This app classifies ECG signals into different types of arrhythmia using a trained **Random Forest & CNN** models.")

if rf_model is not None and label_encoder is not None and cnn_model is not None:

    st.sidebar.header("📌 Choose Input Method")
    option = st.sidebar.radio("How would you like to input data?", 
                             ("Upload CSV", "Enter Manually", "Upload ECG Image"))

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
                    predictions = predict_with_rf(data)
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
            prediction = predict_with_rf(input_df)
            st.success(f"🎯 Predicted Arrhythmia: **{prediction[0]}**")
            st.info(advice.get(prediction[0], "ℹ️ No specific advice available."))

    
    elif option == "Upload ECG Image":
        st.markdown("📤 Upload an image of your ECG signal")
        uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded ECG Image', use_container_width=True)
                
                if st.button("Classify ECG"):
                    with st.spinner("Classifying..."):
                        prediction = predict_with_cnn(image)
                        st.success(f"🎯 Predicted Arrhythmia: **{prediction}**")
                        st.info(advice.get(prediction, "ℹ️ No specific advice available."))
            except Exception as e:
                st.error(f"Error processing image: {e}")

    st.markdown("---")
    st.subheader("ℹ️ About This Model")
    st.write("""
    This app uses two different models for arrhythmia classification:
    
    1. **Random Forest Model**: Classifies based on ECG feature data (CSV/manual input)
    2. **CNN Model**: Classifies based on ECG images
    
    The models classify arrhythmia into:
    - **N**: Normal
    - **Q**: Unknown
    - **SVEB**: Supraventricular ectopic beat
    - **VEB**: Ventricular ectopic beat
    - **F**: Fusion beat

    For clinical use, always consult with a healthcare provider.
    """)
