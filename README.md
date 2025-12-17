# Liver Disease Prediction using Kolmogorov-Arnold Networks (KAN)

This project implements a machine learning model for predicting liver disease using Kolmogorov-Arnold Networks (KAN), a novel neural network architecture. The system includes a web application with a React frontend and a Flask backend for real-time predictions.

## Features

- **KAN Model**: Utilizes Kolmogorov-Arnold Networks for improved prediction accuracy
- **Web Interface**: User-friendly React-based frontend for inputting patient data
- **API Backend**: Flask REST API for model inference
- **Data Visualization**: Generated plots and evaluation metrics
- **Multiple Datasets**: Supports ILPD and Kaggle liver disease datasets

## Project Structure

```
├── backend/                 # Flask API server
│   ├── app.py              # Main API application
│   └── requirements.txt    # Python dependencies
├── frontend/               # React web application
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   ├── main.jsx       # Application entry point
│   │   └── components/
│   │       └── PredictionForm.jsx  # Prediction form component
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/                 # Trained models and metadata
│   ├── kan_model.pth      # Trained KAN model
│   ├── metadata.json      # Model metadata
│   ├── detailed_evaluation_metrics.csv
│   └── model_comparison.csv
├── plots/                  # Generated visualization plots
├── liver_disease_kan.ipynb # Jupyter notebook with model training
├── generate_figures.py    # Script to generate plots
├── ilpd.csv               # Indian Liver Patient Dataset
├── kaggle_liver.csv       # Kaggle liver disease dataset
├── start.ps1              # PowerShell script to start services
├── test_api.ps1           # PowerShell script to test API
├── DEPLOYMENT_README.md   # Deployment instructions
├── HOW_TO_ADD_DIAGRAMS.md # Diagram addition guide
└── IEEE_Conference_Paper.md # Research paper
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- Git

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Usage

### Running the Application

1. Start the backend server:
   ```bash
   cd backend
   python app.py
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:5173`

### Using the PowerShell Scripts

- `start.ps1`: Starts both backend and frontend services
- `test_api.ps1`: Tests the API endpoints

### Model Training

Open `liver_disease_kan.ipynb` in Jupyter Notebook to see the model training process and evaluation.

## API Endpoints

- `POST /predict`: Accepts patient data and returns prediction
  - Input: JSON with patient features
  - Output: Prediction result with confidence score

## Datasets

- **ILPD Dataset**: Indian Liver Patient Dataset
- **Kaggle Dataset**: Liver disease dataset from Kaggle

## Model Performance

The KAN model achieves competitive performance compared to traditional neural networks. Detailed metrics are available in `models/detailed_evaluation_metrics.csv` and `models/model_comparison.csv`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite our IEEE conference paper included in the repository.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
