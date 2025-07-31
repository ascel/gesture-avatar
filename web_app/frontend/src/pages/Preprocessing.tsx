import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  FormControlLabel,
  Switch,
  TextField,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  Settings,
  PlayArrow,
  Refresh,
} from '@mui/icons-material';
import axios from 'axios';

interface PreprocessingConfig {
  normalize: boolean;
  augmentation: boolean;
  noise_level: number;
  rotation_range: number;
}

const Preprocessing: React.FC = () => {
  const [config, setConfig] = useState<PreprocessingConfig>({
    normalize: true,
    augmentation: true,
    noise_level: 0.1,
    rotation_range: 15,
  });

  const [isProcessing, setIsProcessing] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [processedSamples, setProcessedSamples] = useState<number>(0);

  const handleConfigChange = (field: keyof PreprocessingConfig, value: any) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const saveConfig = async () => {
    try {
      await axios.post('/api/preprocessing/configure', config);
      setMessage({ type: 'success', text: 'Configuration saved successfully' });
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to save configuration' });
    }
  };

  const runPreprocessing = async () => {
    setIsProcessing(true);
    setProcessedSamples(0);
    
    try {
      // First save the config
      await axios.post('/api/preprocessing/configure', config);
      
      // Then run preprocessing
      const response = await axios.post('/api/preprocessing/run');
      
      setProcessedSamples(response.data.processed_samples || 0);
      
      if (response.data.error) {
        setMessage({ 
          type: 'info', 
          text: `Preprocessing completed with issues: ${response.data.error}` 
        });
      } else {
        setMessage({ 
          type: 'success', 
          text: `Preprocessing completed! Processed ${response.data.processed_samples} samples.` 
        });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Preprocessing failed' });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Preprocessing
      </Typography>

      {message && (
        <Alert severity={message.type} sx={{ mb: 2 }} onClose={() => setMessage(null)}>
          {message.text}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Preprocessing Configuration
              </Typography>

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Basic Options */}
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Basic Options
                  </Typography>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.normalize}
                        onChange={(e) => handleConfigChange('normalize', e.target.checked)}
                        disabled={isProcessing}
                      />
                    }
                    label="Normalize Data"
                  />
                  
                  <Typography variant="body2" color="textSecondary" sx={{ ml: 4, mb: 2 }}>
                    Apply normalization to improve model training stability
                  </Typography>

                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.augmentation}
                        onChange={(e) => handleConfigChange('augmentation', e.target.checked)}
                        disabled={isProcessing}
                      />
                    }
                    label="Data Augmentation"
                  />
                  
                  <Typography variant="body2" color="textSecondary" sx={{ ml: 4 }}>
                    Apply data augmentation to increase dataset diversity
                  </Typography>
                </Box>

                {/* Augmentation Parameters */}
                {config.augmentation && (
                  <Box>
                    <Typography variant="subtitle1" gutterBottom>
                      Augmentation Parameters
                    </Typography>
                    
                    <TextField
                      fullWidth
                      label="Noise Level"
                      type="number"
                      value={config.noise_level}
                      onChange={(e) => handleConfigChange('noise_level', parseFloat(e.target.value))}
                      disabled={isProcessing}
                      sx={{ mb: 2 }}
                      inputProps={{ min: 0, max: 1, step: 0.01 }}
                      helperText="Amount of noise to add (0.0 - 1.0)"
                    />

                    <TextField
                      fullWidth
                      label="Rotation Range (degrees)"
                      type="number"
                      value={config.rotation_range}
                      onChange={(e) => handleConfigChange('rotation_range', parseInt(e.target.value))}
                      disabled={isProcessing}
                      inputProps={{ min: 0, max: 45 }}
                      helperText="Maximum rotation angle for augmentation"
                    />
                  </Box>
                )}

                {/* Action Buttons */}
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button
                    variant="outlined"
                    startIcon={<Settings />}
                    onClick={saveConfig}
                    disabled={isProcessing}
                  >
                    Save Config
                  </Button>
                  
                  <Button
                    variant="contained"
                    startIcon={<PlayArrow />}
                    onClick={runPreprocessing}
                    disabled={isProcessing}
                  >
                    {isProcessing ? 'Processing...' : 'Run Preprocessing'}
                  </Button>
                </Box>

                {isProcessing && (
                  <Box>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Processing data...
                    </Typography>
                    <LinearProgress />
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Preview/Results Panel */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processing Results
              </Typography>

              {processedSamples > 0 ? (
                <Box>
                  <Typography variant="h4" color="primary" gutterBottom>
                    {processedSamples}
                  </Typography>
                  <Typography variant="body1" color="textSecondary" gutterBottom>
                    Samples processed successfully
                  </Typography>
                  
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Applied Configuration:
                    </Typography>
                    <Box sx={{ pl: 2 }}>
                      <Typography variant="body2">
                        • Normalization: {config.normalize ? 'Enabled' : 'Disabled'}
                      </Typography>
                      <Typography variant="body2">
                        • Augmentation: {config.augmentation ? 'Enabled' : 'Disabled'}
                      </Typography>
                      {config.augmentation && (
                        <>
                          <Typography variant="body2">
                            • Noise Level: {config.noise_level}
                          </Typography>
                          <Typography variant="body2">
                            • Rotation Range: {config.rotation_range}°
                          </Typography>
                        </>
                      )}
                    </Box>
                  </Box>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body1" color="textSecondary">
                    No preprocessing results yet.
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Configure your settings and run preprocessing to see results here.
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Information Card */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                About Preprocessing
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                Data preprocessing is crucial for training robust gesture recognition models. Here's what each option does:
              </Typography>
              
              <Box sx={{ pl: 2 }}>
                <Typography variant="body2" paragraph>
                  <strong>Normalization:</strong> Scales data to a consistent range, improving training stability and convergence speed.
                </Typography>
                
                <Typography variant="body2" paragraph>
                  <strong>Data Augmentation:</strong> Creates variations of existing samples to increase dataset diversity and reduce overfitting.
                </Typography>
                
                <Typography variant="body2" paragraph>
                  <strong>Noise Addition:</strong> Adds controlled noise to make the model more robust to real-world variations.
                </Typography>
                
                <Typography variant="body2">
                  <strong>Rotation:</strong> Applies random rotations to simulate different hand orientations.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Preprocessing; 