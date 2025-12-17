import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define KAN Model Architecture (matching notebook implementation exactly)
class SplineActivation(torch.nn.Module):
    def __init__(self, input_dim, n_basis=8):
        super().__init__()
        self.input_dim = input_dim
        self.n_basis = n_basis
        # Learnable parameters for Gaussian RBF (separate for each input feature)
        self.centers = torch.nn.Parameter(torch.linspace(-3, 3, n_basis).unsqueeze(0).repeat(input_dim, 1))
        width_val = (6 / (n_basis - 1)) if n_basis > 1 else 1.0
        self.widths = torch.nn.Parameter(torch.ones(input_dim, n_basis) * width_val)
        self.weights = torch.nn.Parameter(torch.randn(input_dim, n_basis) * 0.01)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        centers = self.centers.unsqueeze(0)  # (1, input_dim, n_basis)
        widths = self.widths.unsqueeze(0)  # (1, input_dim, n_basis)
        weights = self.weights.unsqueeze(0)  # (1, input_dim, n_basis)

        # Gaussian RBF
        rbf = torch.exp(-0.5 * ((x - centers) / widths) ** 2)  # (batch_size, input_dim, n_basis)
        output = (rbf * weights).sum(dim=-1)  # (batch_size, input_dim)
        return output

class KANLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_basis=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_basis = n_basis

        # Spline activation for all features
        self.spline_activation = SplineActivation(input_dim, n_basis)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Apply spline activation to all features at once
        x_splined = self.spline_activation(x)  # (batch_size, input_dim)
        output = self.linear(x_splined)  # (batch_size, output_dim)
        return output

class KANModel(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, n_layers=3, n_basis=8, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_basis = n_basis

        # Build KAN layers
        layers = []

        # First KAN layer: input_dim -> hidden_dim
        layers.append(KANLayer(input_dim, hidden_dim, n_basis))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))

        # Hidden KAN layers: hidden_dim -> hidden_dim
        for _ in range(n_layers - 1):
            layers.append(KANLayer(hidden_dim, hidden_dim, n_basis))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))

        # Combine all layers
        self.kan_layers = torch.nn.Sequential(*layers)

        # Final output layer: hidden_dim -> 1 (binary classification)
        self.output_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Pass through KAN layers
        x = self.kan_layers(x)

        # Final output layer
        logits = self.output_layer(x)

        return logits

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    MODEL_DIR = Path("models")
    device = torch.device('cpu')

    try:
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")

        # Load KAN model
        model = KANModel(input_dim=10, hidden_dim=64, n_layers=3, n_basis=8, dropout=0.2)
        checkpoint = torch.load(MODEL_DIR / "kan_model.pth", map_location=device)

        # Handle different save formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        logger.info("✓ KAN model and scaler loaded successfully")
        return model, scaler
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
    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
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

        # Make prediction
        with torch.no_grad():
            x_tensor = torch.FloatTensor(features_scaled)
            logits = model(x_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            probability = float(probabilities[0])
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