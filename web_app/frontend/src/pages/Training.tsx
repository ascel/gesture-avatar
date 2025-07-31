import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  School,
  Stop,
  Refresh,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import axios from 'axios';

interface TrainingConfig {
  model_backbone: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  validation_split: number;
}

interface TrainingSession {
  status: string;
  config: TrainingConfig;
  start_time: string;
  end_time?: string;
  progress: number;
  current_epoch: number;
  metrics: {
    train_loss: number[];
    train_accuracy: number[];
    val_loss: number[];
    val_accuracy: number[];
  };
  final_accuracy?: number;
  model_path?: string;
  error?: string;
}

const Training: React.FC = () => {
  const [config, setConfig] = useState<TrainingConfig>({
    model_backbone: 'resnet',
    epochs: 50,
    batch_size: 32,
    learning_rate: 0.001,
    validation_split: 0.2,
  });

  const [currentSession, setCurrentSession] = useState<string | null>(null);
  const [sessionData, setSessionData] = useState<TrainingSession | null>(null);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (currentSession && isTraining) {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`/api/training/status/${currentSession}`);
          setSessionData(response.data);
          
          if (response.data.status === 'completed' || response.data.status === 'failed') {
            setIsTraining(false);
            if (response.data.status === 'completed') {
              setMessage({ 
                type: 'success', 
                text: `Training completed! Final accuracy: ${(response.data.final_accuracy * 100).toFixed(2)}%` 
              });
            } else {
              setMessage({ 
                type: 'error', 
                text: `Training failed: ${response.data.error}` 
              });
            }
          }
        } catch (error) {
          console.error('Error fetching training status:', error);
        }
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [currentSession, isTraining]);

  const startTraining = async () => {
    try {
      const response = await axios.post('/api/training/start', config);
      setCurrentSession(response.data.session_id);
      setIsTraining(true);
      setSessionData(null);
      setMessage({ type: 'info', text: 'Training started...' });
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to start training' });
    }
  };

  const prepareChartData = () => {
    if (!sessionData?.metrics) return [];
    
    const data = [];
    const maxLength = Math.max(
      sessionData.metrics.train_loss.length,
      sessionData.metrics.val_loss.length
    );

    for (let i = 0; i < maxLength; i++) {
      data.push({
        epoch: i + 1,
        train_loss: sessionData.metrics.train_loss[i] || null,
        train_accuracy: sessionData.metrics.train_accuracy[i] ? sessionData.metrics.train_accuracy[i] * 100 : null,
        val_loss: sessionData.metrics.val_loss[i] || null,
        val_accuracy: sessionData.metrics.val_accuracy[i] ? sessionData.metrics.val_accuracy[i] * 100 : null,
      });
    }

    return data;
  };

  const chartData = prepareChartData();

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Training
      </Typography>

      {message && (
        <Alert severity={message.type} sx={{ mb: 2 }} onClose={() => setMessage(null)}>
          {message.text}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Training Configuration */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Configuration
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Model Backbone</InputLabel>
                <Select
                  value={config.model_backbone}
                  label="Model Backbone"
                  onChange={(e) => setConfig({ ...config, model_backbone: e.target.value })}
                  disabled={isTraining}
                >
                  <MenuItem value="resnet">ResNet</MenuItem>
                  <MenuItem value="efficientnet">EfficientNet</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="Epochs"
                type="number"
                value={config.epochs}
                onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                disabled={isTraining}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, max: 200 }}
              />

              <TextField
                fullWidth
                label="Batch Size"
                type="number"
                value={config.batch_size}
                onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                disabled={isTraining}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, max: 128 }}
              />

              <TextField
                fullWidth
                label="Learning Rate"
                type="number"
                value={config.learning_rate}
                onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
                disabled={isTraining}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.0001, max: 0.1, step: 0.0001 }}
              />

              <TextField
                fullWidth
                label="Validation Split"
                type="number"
                value={config.validation_split}
                onChange={(e) => setConfig({ ...config, validation_split: parseFloat(e.target.value) })}
                disabled={isTraining}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.1, max: 0.5, step: 0.05 }}
              />

              <Button
                fullWidth
                variant="contained"
                startIcon={<School />}
                onClick={startTraining}
                disabled={isTraining}
                size="large"
              >
                {isTraining ? 'Training...' : 'Start Training'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Training Progress */}
        <Grid item xs={12} md={8}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Progress
              </Typography>

              {sessionData ? (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="body1">
                      Status: <Chip 
                        label={sessionData.status.toUpperCase()} 
                        color={
                          sessionData.status === 'completed' ? 'success' : 
                          sessionData.status === 'failed' ? 'error' : 
                          sessionData.status === 'running' ? 'primary' : 'default'
                        }
                      />
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Epoch {sessionData.current_epoch} / {sessionData.config.epochs}
                    </Typography>
                  </Box>

                  <LinearProgress 
                    variant="determinate" 
                    value={sessionData.progress} 
                    sx={{ mb: 2, height: 8, borderRadius: 4 }}
                  />

                  <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                    Progress: {sessionData.progress.toFixed(1)}%
                  </Typography>

                  {sessionData.metrics.val_accuracy.length > 0 && (
                    <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                      <Typography variant="body2">
                        Latest Train Accuracy: {(sessionData.metrics.train_accuracy[sessionData.metrics.train_accuracy.length - 1] * 100).toFixed(2)}%
                      </Typography>
                      <Typography variant="body2">
                        Latest Val Accuracy: {(sessionData.metrics.val_accuracy[sessionData.metrics.val_accuracy.length - 1] * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                  )}
                </Box>
              ) : (
                <Typography variant="body2" color="textSecondary">
                  No training session active
                </Typography>
              )}
            </CardContent>
          </Card>

          {/* Training Metrics Charts */}
          {chartData.length > 0 && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Training Metrics
                </Typography>

                <Grid container spacing={2}>
                  {/* Accuracy Chart */}
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Accuracy
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="epoch" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip formatter={(value: any) => [`${value?.toFixed(2)}%`, '']} />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="train_accuracy" 
                          stroke="#2196f3" 
                          name="Training Accuracy"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="val_accuracy" 
                          stroke="#ff9800" 
                          name="Validation Accuracy"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </Grid>

                  {/* Loss Chart */}
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Loss
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="epoch" />
                        <YAxis />
                        <Tooltip formatter={(value: any) => [value?.toFixed(4), '']} />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="train_loss" 
                          stroke="#4caf50" 
                          name="Training Loss"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="val_loss" 
                          stroke="#f44336" 
                          name="Validation Loss"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default Training; 