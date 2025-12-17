import streamlit as st
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    MODEL_DIR = Path("models")
    device = 'cpu'

    try:
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")

        # Load ONNX model
        onnx_model_path = MODEL_DIR / "kan_model.onnx"
        session = ort.InferenceSession(str(onnx_model_path))

        logger.info("✓ ONNX model and scaler loaded successfully")
        return session, scaler
    except Exception as e:
        logger.error(f"✗ Failed to load model or scaler: {e}")
        st.error(f"Failed to load model: {e}")
        return None, None

def main():
    st.title("Liver Disease Prediction using KAN")
    st.markdown("""
    This app predicts the likelihood of liver disease based on clinical features using a Kolmogorov-Arnold Network (KAN) model.
    """)

    # Load model and scaler
    session, scaler = load_model_and_scaler()

    if session is None or scaler is None:
        st.error("Model could not be loaded. Please check the models directory.")
        return

    st.header("Patient Information")

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
        gender_val = gender[1]
        total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, value=0.9)
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, value=0.3)
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase (IU/L)", min_value=0, value=202)

    with col2:
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT) (IU/L)", min_value=0, value=32)
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST) (IU/L)", min_value=0, value=35)
        total_proteins = st.number_input("Total Proteins (g/dL)", min_value=0.0, value=7.2)
        albumin = st.number_input("Albumin (g/dL)", min_value=0.0, value=4.1)
        albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.32)

    if st.button("Predict Liver Disease"):
        # Prepare data
        features = pd.DataFrame([[age, gender_val, total_bilirubin, direct_bilirubin,
                                  alkaline_phosphotase, alamine_aminotransferase,
                                  aspartate_aminotransferase, total_proteins, albumin,
                                  albumin_and_globulin_ratio]],
                                columns=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                                         'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                                         'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
                                         'Albumin_and_Globulin_Ratio'])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction using ONNX
        # Convert to numpy array and ensure correct shape
        x_numpy = np.array(features_scaled, dtype=np.float32)

        # Run inference
        ort_inputs = {session.get_inputs()[0].name: x_numpy}
        logits = session.run(None, ort_inputs)[0]

        # Apply sigmoid to get probabilities
        probabilities = 1 / (1 + np.exp(-logits))  # sigmoid
        probability = float(probabilities.flatten()[0])  # Probability of disease
        prediction = int(probability >= 0.5)
        confidence = float(max(probability, 1 - probability))

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
            message = "Low probability of liver disease. Maintain healthy lifestyle."
            color = "green"
        elif probability < 0.6:
            risk_level = "Moderate Risk"
            message = "Moderate risk detected. Consider consulting a healthcare provider."
            color = "orange"
        elif probability < 0.8:
            risk_level = "High Risk"
            message = "High risk of liver disease. Medical consultation recommended."
            color = "red"
        else:
            risk_level = "Very High Risk"
            message = "Very high risk detected. Immediate medical attention advised."
            color = "darkred"

        st.header("Prediction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", "Disease" if prediction == 1 else "No Disease")
        with col2:
            st.metric("Probability", f"{probability:.1%}")
        with col3:
            st.metric("Confidence", f"{confidence:.1%}")

        st.subheader(f"Risk Level: {risk_level}")
        st.markdown(f"<p style='color:{color}; font-size:18px;'>{message}</p>", unsafe_allow_html=True)

        # Display input summary
        st.subheader("Input Summary")
        st.json({
            "age": age,
            "gender": "Male" if gender_val == 1 else "Female",
            "total_bilirubin": total_bilirubin,
            "direct_bilirubin": direct_bilirubin,
            "alkaline_phosphotase": alkaline_phosphotase,
            "alamine_aminotransferase": alamine_aminotransferase,
            "aspartate_aminotransferase": aspartate_aminotransferase,
            "total_proteins": total_proteins,
            "albumin": albumin,
            "albumin_and_globulin_ratio": albumin_and_globulin_ratio
        })

if __name__ == "__main__":
    main()