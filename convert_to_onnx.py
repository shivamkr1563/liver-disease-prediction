import torch
import numpy as np
from pathlib import Path
import joblib

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

def convert_to_onnx():
    MODEL_DIR = Path("models")

    # Load scaler
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    # Load KAN model
    model = KANModel(input_dim=10, hidden_dim=64, n_layers=3, n_basis=8, dropout=0.2)
    checkpoint = torch.load(MODEL_DIR / "kan_model.pth", map_location='cpu')

    # Handle different save formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 10)  # Batch size 1, 10 features

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_DIR / "kan_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("Model converted to ONNX format successfully!")
    print(f"ONNX model saved to: {MODEL_DIR / 'kan_model.onnx'}")

if __name__ == "__main__":
    convert_to_onnx()