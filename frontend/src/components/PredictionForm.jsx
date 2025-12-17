import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Grid,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Divider,
  Chip
} from '@mui/material';
import {
  LocalHospital,
  Science,
  TrendingUp,
  CheckCircle,
  Warning,
  Error as ErrorIcon
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

function PredictionForm() {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    total_bilirubin: '',
    direct_bilirubin: '',
    alkaline_phosphotase: '',
    alamine_aminotransferase: '',
    aspartate_aminotransferase: '',
    total_proteins: '',
    albumin: '',
    albumin_and_globulin_ratio: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Convert form data to numbers
      const payload = {
        age: parseFloat(formData.age),
        gender: parseInt(formData.gender),
        total_bilirubin: parseFloat(formData.total_bilirubin),
        direct_bilirubin: parseFloat(formData.direct_bilirubin),
        alkaline_phosphotase: parseFloat(formData.alkaline_phosphotase),
        alamine_aminotransferase: parseFloat(formData.alamine_aminotransferase),
        aspartate_aminotransferase: parseFloat(formData.aspartate_aminotransferase),
        total_proteins: parseFloat(formData.total_proteins),
        albumin: parseFloat(formData.albumin),
        albumin_and_globulin_ratio: parseFloat(formData.albumin_and_globulin_ratio)
      };

      const response = await axios.post(`${API_BASE_URL}/predict`, payload);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction. Please check your inputs.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      age: '',
      gender: '',
      total_bilirubin: '',
      direct_bilirubin: '',
      alkaline_phosphotase: '',
      alamine_aminotransferase: '',
      aspartate_aminotransferase: '',
      total_proteins: '',
      albumin: '',
      albumin_and_globulin_ratio: ''
    });
    setResult(null);
    setError(null);
  };

  const getRiskColor = (riskLevel) => {
    if (riskLevel === 'Low Risk') return 'success';
    if (riskLevel === 'Moderate Risk') return 'warning';
    if (riskLevel === 'High Risk') return 'error';
    if (riskLevel === 'Very High Risk') return 'error';
    return 'default';
  };

  const getRiskIcon = (riskLevel) => {
    if (riskLevel === 'Low Risk') return <CheckCircle />;
    if (riskLevel === 'Moderate Risk') return <Warning />;
    return <ErrorIcon />;
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <LocalHospital sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
        <Typography variant="h3" component="h1" gutterBottom fontWeight="bold">
          Liver Disease Prediction
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          AI-Powered Clinical Decision Support System
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Using KAN (Kolmogorov-Arnold Network) Deep Learning Model (95.57% ROC-AUC)
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={7}>
          <Paper elevation={3} sx={{ p: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Science sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h5" component="h2" fontWeight="bold">
                Patient Clinical Data
              </Typography>
            </Box>

            <form onSubmit={handleSubmit}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Age"
                    name="age"
                    type="number"
                    value={formData.age}
                    onChange={handleChange}
                    required
                    inputProps={{ min: 1, max: 120, step: 1 }}
                    helperText="Patient age in years"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth required>
                    <InputLabel>Gender</InputLabel>
                    <Select
                      name="gender"
                      value={formData.gender}
                      onChange={handleChange}
                      label="Gender"
                    >
                      <MenuItem value={0}>Female</MenuItem>
                      <MenuItem value={1}>Male</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Total Bilirubin"
                    name="total_bilirubin"
                    type="number"
                    value={formData.total_bilirubin}
                    onChange={handleChange}
                    required
                    inputProps={{ min: 0, step: 0.1 }}
                    helperText="mg/dL (Normal: 0.1-1.2)"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Direct Bilirubin"
                    name="direct_bilirubin"
                    type="number"
                    value={formData.direct_bilirubin}
                    onChange={handleChange}
                    required
                    inputProps={{ min: 0, step: 0.1 }}
                    helperText="mg/dL (Normal: 0-0.3)"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Alkaline Phosphotase"
                    name="alkaline_phosphotase"
                    type="number"
                    value={formData.alkaline_phosphotase}
                    onChange={handleChange}
                    required
                    inputProps={{ min: 0, step: 1 }}
                    helperText="IU/L (Normal: 44-147)"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="ALT (Alamine Aminotransferase)"
                    name="alamine_aminotransferase"
                    type="number"
                    value={formData.alamine_aminotransferase}
                    onChange={handleChange}
                    required
                    inputProps={{ min: 0, step: 1 }}
                    helperText="IU/L (Normal: 7-56)"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="AST (Aspartate Aminotransferase)"
                    name="aspartate_aminotransferase"
                    type="number"
                    value={formData.aspartate_aminotransferase}
                    onChange={handleChange}
                    required
                    inputProps={{ min: 0, step: 1 }}
                    helperText="IU/L (Normal: 10-40)"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Total Proteins"
                    name="total_proteins"
                    type="number"
                    value={formData.total_proteins}
                    onChange={handleChange}
                    required
                    inputProps={{ min: 0, step: 0.1 }}
                    helperText="g/dL (Normal: 6.0-8.3)"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Albumin"
                    name="albumin"
                    type="number"
                    value={formData.albumin}
                    onChange={handleChange}
                    required
                    inputProps={{ min: 0, step: 0.1 }}
                    helperText="g/dL (Normal: 3.5-5.5)"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Albumin/Globulin Ratio"
                    name="albumin_and_globulin_ratio"
                    type="number"
                    value={formData.albumin_and_globulin_ratio}
                    onChange={handleChange}
                    required
                    inputProps={{ min: 0, step: 0.01 }}
                    helperText="Ratio (Normal: 1.0-2.5)"
                  />
                </Grid>
              </Grid>

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  disabled={loading}
                  fullWidth
                  startIcon={loading ? <CircularProgress size={20} /> : <TrendingUp />}
                >
                  {loading ? 'Analyzing...' : 'Predict'}
                </Button>
                <Button
                  type="button"
                  variant="outlined"
                  size="large"
                  onClick={handleReset}
                  disabled={loading}
                >
                  Reset
                </Button>
              </Box>
            </form>
          </Paper>
        </Grid>

        <Grid item xs={12} md={5}>
          <Box sx={{ position: 'sticky', top: 20 }}>
            {error && (
              <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
                {error}
              </Alert>
            )}

            {result && (
              <Card elevation={3}>
                <CardContent>
                  <Box sx={{ textAlign: 'center', mb: 2 }}>
                    <Typography variant="h5" gutterBottom fontWeight="bold">
                      Prediction Result
                    </Typography>
                    <Divider sx={{ my: 2 }} />
                  </Box>

                  <Box sx={{ mb: 3, textAlign: 'center' }}>
                    <Chip
                      icon={getRiskIcon(result.risk_level)}
                      label={result.risk_level}
                      color={getRiskColor(result.risk_level)}
                      sx={{ 
                        fontSize: '1.1rem', 
                        py: 2.5, 
                        px: 2,
                        fontWeight: 'bold'
                      }}
                    />
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Diagnosis
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      {result.prediction === 1 ? 'Liver Disease Detected' : 'No Liver Disease'}
                    </Typography>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Disease Probability
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      {(result.probability * 100).toFixed(2)}%
                    </Typography>
                    <Box
                      sx={{
                        width: '100%',
                        height: 10,
                        bgcolor: 'grey.200',
                        borderRadius: 1,
                        overflow: 'hidden',
                        mt: 1
                      }}
                    >
                      <Box
                        sx={{
                          width: `${result.probability * 100}%`,
                          height: '100%',
                          bgcolor: getRiskColor(result.risk_level) + '.main',
                          transition: 'width 0.5s ease'
                        }}
                      />
                    </Box>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Model Confidence
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      {(result.confidence * 100).toFixed(2)}%
                    </Typography>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  <Alert 
                    severity={getRiskColor(result.risk_level)} 
                    sx={{ mt: 2 }}
                    icon={getRiskIcon(result.risk_level)}
                  >
                    <Typography variant="body2">
                      {result.message}
                    </Typography>
                  </Alert>

                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                    * This prediction is for informational purposes only and should not replace 
                    professional medical advice, diagnosis, or treatment.
                  </Typography>
                </CardContent>
              </Card>
            )}

            {!result && !error && (
              <Card elevation={3} sx={{ bgcolor: 'grey.50' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom fontWeight="bold">
                    ℹ️ Instructions
                  </Typography>
                  <Typography variant="body2" paragraph>
                    1. Enter all clinical parameters in the form
                  </Typography>
                  <Typography variant="body2" paragraph>
                    2. Ensure values are within normal physiological ranges
                  </Typography>
                  <Typography variant="body2" paragraph>
                    3. Click "Predict" to get AI-powered analysis
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    The KAN model uses 10 clinical features with learnable spline activations 
                    to predict liver disease (95.57% ROC-AUC on test data).
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Box>
        </Grid>
      </Grid>
    </Container>
  );
}

export default PredictionForm;
