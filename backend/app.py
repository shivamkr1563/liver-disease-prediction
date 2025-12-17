"""
FastAPI Backend for Liver Disease Prediction
Uses the trained KAN (Kolmogorov-Arnold Network) model for predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Liver Disease Prediction API (KAN)",
    description="API for predicting liver disease using KAN neural network with clinical features",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Load trained model and scaler
MODEL_DIR = Path(__file__).parent.parent / "models"
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
except Exception as e:
    logger.error(f"✗ Failed to load model or scaler: {e}")
    raise


# Input data model
class PatientData(BaseModel):
    """Patient clinical features for liver disease prediction"""
    
    age: float = Field(..., ge=1, le=120, description="Patient age in years")
    gender: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    total_bilirubin: float = Field(..., ge=0, description="Total Bilirubin (mg/dL)")
    direct_bilirubin: float = Field(..., ge=0, description="Direct Bilirubin (mg/dL)")
    alkaline_phosphotase: float = Field(..., ge=0, description="Alkaline Phosphotase (IU/L)")
    alamine_aminotransferase: float = Field(..., ge=0, description="Alamine Aminotransferase (ALT) (IU/L)")
    aspartate_aminotransferase: float = Field(..., ge=0, description="Aspartate Aminotransferase (AST) (IU/L)")
    total_proteins: float = Field(..., ge=0, description="Total Proteins (g/dL)")
    albumin: float = Field(..., ge=0, description="Albumin (g/dL)")
    albumin_and_globulin_ratio: float = Field(..., ge=0, description="Albumin and Globulin Ratio")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "gender": 1,
                "total_bilirubin": 0.9,
                "direct_bilirubin": 0.3,
                "alkaline_phosphotase": 202,
                "alamine_aminotransferase": 32,
                "aspartate_aminotransferase": 35,
                "total_proteins": 7.2,
                "albumin": 4.1,
                "albumin_and_globulin_ratio": 1.32
            }
        }


# Response model
class PredictionResponse(BaseModel):
    """Prediction response with probability and risk level"""
    
    prediction: int = Field(..., description="Predicted class (0=No Disease, 1=Disease)")
    probability: float = Field(..., description="Probability of having liver disease")
    risk_level: str = Field(..., description="Risk level category")
    confidence: float = Field(..., description="Model confidence (max probability)")
    message: str = Field(..., description="Human-readable message")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Liver Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """
    Predict liver disease based on patient clinical features
    
    Returns:
        PredictionResponse with prediction, probability, and risk level
    """
    try:
        # Convert input to DataFrame with correct feature order
        features = pd.DataFrame([[
            patient_data.age,
            patient_data.gender,
            patient_data.total_bilirubin,
            patient_data.direct_bilirubin,
            patient_data.alkaline_phosphotase,
            patient_data.alamine_aminotransferase,
            patient_data.aspartate_aminotransferase,
            patient_data.total_proteins,
            patient_data.albumin,
            patient_data.albumin_and_globulin_ratio
        ]], columns=[
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
            'Albumin_and_Globulin_Ratio'
        ])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Convert to tensor and make prediction
        with torch.no_grad():
            x_tensor = torch.FloatTensor(features_scaled)
            logits = model(x_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            probability = float(probabilities[0])  # Probability of disease
            prediction = int(probability >= 0.5)
            confidence = float(max(probability, 1 - probability))
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
            message = "Low probability of liver disease. Maintain healthy lifestyle."
        elif probability < 0.6:
            risk_level = "Moderate Risk"
            message = "Moderate risk detected. Consider consulting a healthcare provider."
        elif probability < 0.8:
            risk_level = "High Risk"
            message = "High risk of liver disease. Medical consultation recommended."
        else:
            risk_level = "Very High Risk"
            message = "Very high risk detected. Immediate medical attention advised."
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Risk: {risk_level}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
