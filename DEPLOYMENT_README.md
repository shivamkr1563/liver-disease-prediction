# Liver Disease Prediction - Web Application Deployment

A full-stack web application for liver disease prediction using a trained Random Forest machine learning model (99.89% ROC-AUC).

## 🏗️ Architecture

- **Backend**: FastAPI (Python) - REST API with ML model
- **Frontend**: React + Vite + Material-UI - Modern responsive interface
- **Model**: Random Forest Classifier (scikit-learn)
- **Dataset**: 31,274 clinical records

## 📁 Project Structure

```
Liver Disease Prediction(KAN)/
├── backend/
│   ├── app.py                    # FastAPI application
│   └── requirements.txt          # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── PredictionForm.jsx   # Main form component
│   │   ├── App.jsx               # App wrapper with theme
│   │   └── main.jsx              # Entry point
│   ├── index.html                # HTML template
│   ├── package.json              # Node dependencies
│   └── vite.config.js            # Vite configuration
├── models/
│   ├── random_forest.pkl         # Trained model
│   └── scaler.pkl                # StandardScaler
└── datasets/                     # Training data
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18 or higher
- **npm**: 9 or higher

### Backend Setup

1. Navigate to the backend directory:
```powershell
cd backend
```

2. Install Python dependencies:
```powershell
pip install -r requirements.txt
```

3. Run the FastAPI server:
```powershell
python app.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
```powershell
cd frontend
```

2. Install Node dependencies:
```powershell
npm install
```

3. Start the development server:
```powershell
npm run dev
```

The app will open automatically at `http://localhost:3000`

## 🔌 API Endpoints

### Base URL: `http://localhost:8000`

#### 1. Root
```
GET /
```
Returns API information and available endpoints.

#### 2. Health Check
```
GET /health
```
Returns server health status and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

#### 3. Predict
```
POST /predict
```
Make a liver disease prediction based on clinical features.

**Request Body:**
```json
{
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
```

**Parameters:**
- `age`: Patient age (1-120 years)
- `gender`: 0 = Female, 1 = Male
- `total_bilirubin`: mg/dL (Normal: 0.1-1.2)
- `direct_bilirubin`: mg/dL (Normal: 0-0.3)
- `alkaline_phosphotase`: IU/L (Normal: 44-147)
- `alamine_aminotransferase`: ALT in IU/L (Normal: 7-56)
- `aspartate_aminotransferase`: AST in IU/L (Normal: 10-40)
- `total_proteins`: g/dL (Normal: 6.0-8.3)
- `albumin`: g/dL (Normal: 3.5-5.5)
- `albumin_and_globulin_ratio`: Ratio (Normal: 1.0-2.5)

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "Low Risk",
  "confidence": 0.85,
  "message": "Low probability of liver disease. Maintain healthy lifestyle."
}
```

**Risk Levels:**
- **Low Risk**: Probability < 30%
- **Moderate Risk**: Probability 30-60%
- **High Risk**: Probability 60-80%
- **Very High Risk**: Probability > 80%

## 🧪 Testing the API

### Using cURL

```powershell
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
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
  }'
```

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make a prediction
data = {
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

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## 🖥️ Frontend Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Validation**: Input validation with helpful hints
- **Visual Feedback**: Color-coded risk levels with progress bars
- **Loading States**: User-friendly loading indicators
- **Error Handling**: Clear error messages
- **Material-UI**: Modern, professional interface

## 📊 Model Performance

- **Model**: Random Forest Classifier
- **Training Data**: 31,274 samples
- **Features**: 10 clinical parameters
- **Performance Metrics**:
  - ROC-AUC: 99.89%
  - Accuracy: 99.20%
  - Precision: 99.51%
  - Recall: 99.51%
  - F1 Score: 99.51%

## 🛠️ Development

### Backend Development

To enable auto-reload during development:
```powershell
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```powershell
cd frontend
npm run dev
```

### Building for Production

**Frontend:**
```powershell
cd frontend
npm run build
npm run preview
```

## 📝 API Documentation

Once the backend is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 🔒 Security Notes

- This is a development setup. For production:
  - Enable HTTPS
  - Add authentication/authorization
  - Implement rate limiting
  - Use environment variables for configuration
  - Add input sanitization
  - Deploy behind a reverse proxy

## ⚕️ Medical Disclaimer

**IMPORTANT**: This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.

## 📦 Dependencies

### Backend
- fastapi==0.104.1
- uvicorn==0.24.0
- scikit-learn==1.3.2
- pandas==2.1.3
- numpy==1.26.2
- pydantic==2.5.0
- joblib==1.3.2

### Frontend
- react==18.2.0
- @mui/material==5.14.20
- axios==1.6.2
- vite==5.0.8

## 🐛 Troubleshooting

### Backend Issues

**Problem**: Module not found errors
```powershell
pip install -r requirements.txt --upgrade
```

**Problem**: Model file not found
- Ensure `models/random_forest.pkl` and `models/scaler.pkl` exist
- Check that you're running from the correct directory

### Frontend Issues

**Problem**: Cannot connect to API
- Ensure backend is running on port 8000
- Check CORS settings in `backend/app.py`

**Problem**: Build errors
```powershell
rm -rf node_modules package-lock.json
npm install
```

## 📧 Support

For issues or questions, please refer to the main project documentation or notebook files.

## 🎓 Citation

If you use this system in your research, please cite:
```
Liver Disease Prediction Using Kolmogorov-Arnold Networks (KAN)
Machine Learning System with Random Forest Classifier
Dataset: Indian Liver Patient Dataset (ILPD) + Kaggle Liver Disease Data
```

## 📄 License

This project is for educational purposes. Please consult with appropriate medical and legal professionals before any clinical use.

---

**Built with ❤️ using FastAPI, React, and scikit-learn**
