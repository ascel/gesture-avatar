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
  current_metrics?: {
    train_loss: number;
    train_accuracy: number;
    val_loss: number;
    val_accuracy: number;
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
  const [activeWebSocket, setActiveWebSocket] = useState<WebSocket | null>(null);

  // Function to establish WebSocket connection immediately
  const connectWebSocket = (sessionId: string) => {
    // Close existing connection if any
    if (activeWebSocket) {
      console.log('🔌 Closing existing WebSocket connection');
      activeWebSocket.close();
      setActiveWebSocket(null);
    }
    
    console.log(`🚀 Immediately connecting WebSocket for session: ${sessionId}`);
    
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname;
      const port = process.env.NODE_ENV === 'development' ? '8000' : window.location.port;
      const wsUrl = `${protocol}//${host}:${port}/ws/training/${sessionId}`;
      
      console.log(`🔗 Immediate WebSocket connection to: ${wsUrl}`);
      console.log('🔍 Connection details:');
      console.log('  - Protocol:', protocol);
      console.log('  - Host:', host);
      console.log('  - Port:', port);
      console.log('  - Session ID:', sessionId);
      console.log('  - NODE_ENV:', process.env.NODE_ENV);
      console.log('  - Current location:', window.location.href);
      
      const websocket = new WebSocket(wsUrl);
      // Mark active immediately to avoid duplicate connection via useEffect
      setActiveWebSocket(websocket);
      
      websocket.onopen = () => {
        console.log('✅ Immediate WebSocket connected successfully!');
        console.log('🔍 WebSocket readyState:', websocket?.readyState);
        console.log('🔍 WebSocket URL:', websocket?.url);
        // Do not set a global toast here; avoid flicker with completion message
        
        // Test if WebSocket can receive data by checking session status immediately
        console.log('🧪 Testing WebSocket - should receive initial session data within 1-2 seconds...');
        setTimeout(() => {
          if (websocket?.readyState === WebSocket.OPEN) {
            console.log('🔍 WebSocket still open after 2 seconds, checking if we received any data...');
          }
        }, 2000);
      };
      
      websocket.onmessage = (event) => {
        console.log('📨 Raw WebSocket message received:', event);
        console.log('📨 Message data type:', typeof event.data);
        console.log('📨 Message data:', event.data);
        
        try {
          const data = JSON.parse(event.data);
          console.log('📡 Parsed WebSocket update received:', data);
          
          // Enhanced logging for training progress
          if (data.status === 'running' && data.current_epoch) {
            console.log(`🔄 Training Progress: Epoch ${data.current_epoch}/${data.config?.epochs || '?'} (${data.progress?.toFixed(1) || 0}%)`);
            if (data.current_metrics) {
              const metrics = data.current_metrics;
              console.log(`📊 Current Metrics: Loss=${metrics.train_loss?.toFixed(4) || 'N/A'}, Acc=${(metrics.train_accuracy * 100)?.toFixed(2) || 'N/A'}%, Val_Loss=${metrics.val_loss?.toFixed(4) || 'N/A'}, Val_Acc=${(metrics.val_accuracy * 100)?.toFixed(2) || 'N/A'}%`);
            }
          }
          
          setSessionData(data);
          
          if (data.status === 'completed' || data.status === 'failed') {
            setIsTraining(false);
            if (data.status === 'completed') {
              console.log('✅ Training completed successfully!');
              setMessage({ 
                type: 'success', 
                text: `Training completed! Final accuracy: ${(data.final_accuracy * 100).toFixed(2)}%` 
              });
            } else {
              console.log('❌ Training failed:', data.error);
              setMessage({ 
                type: 'error', 
                text: `Training failed: ${data.error}` 
              });
            }
            websocket?.close();
            setActiveWebSocket(null);
          } else if (data.status === 'running') {
            setIsTraining(true);
          }
        } catch (error) {
          console.error('❌ Failed to parse immediate WebSocket message:', error);
          console.error('❌ Raw message that failed:', event.data);
        }
      };
      
      websocket.onerror = (error) => {
        console.error('❌ Immediate WebSocket error:', error);
        console.log('🔍 WebSocket readyState on error:', websocket?.readyState);
        console.log('🔍 WebSocket URL on error:', websocket?.url);
        setMessage({ type: 'error', text: 'WebSocket connection failed - please check backend is running' });
        setActiveWebSocket(null);
      };
      
      websocket.onclose = (event) => {
        console.log('🔌 Immediate WebSocket connection closed');
        console.log('🔍 Close event code:', event.code);
        console.log('🔍 Close event reason:', event.reason);
        console.log('🔍 Close event wasClean:', event.wasClean);
        setActiveWebSocket(null);
      };
      
      return websocket;
    } catch (error) {
      console.error('Failed to create immediate WebSocket:', error);
      return null;
    }
  };

  useEffect(() => {
    let websocket: WebSocket | null = null;
    
    if (currentSession) {
      console.log(`🔌 useEffect triggered for session: ${currentSession}`);
      
      // Only create WebSocket if no active connection exists
      if (!activeWebSocket) {
        console.log('📡 No active WebSocket found, creating new connection...');
        
        // Try WebSocket for real-time updates
        try {
          // Use relative WebSocket URL to work with proxy
          const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
          const host = window.location.hostname;
          const port = process.env.NODE_ENV === 'development' ? '8000' : window.location.port;
          const wsUrl = `${protocol}//${host}:${port}/ws/training/${currentSession}`;
          
          console.log(`🔗 useEffect WebSocket connection to: ${wsUrl}`);
          websocket = new WebSocket(wsUrl);
          // Mark active immediately to avoid race with the immediate connector
          setActiveWebSocket(websocket);
          
          websocket.onopen = () => {
            // Only log connection established
            console.log('✅ WebSocket connected for real-time training updates');
            // Avoid setting a global message to prevent flicker
          };
        
          websocket.onmessage = (event) => {
            try {
              const data: any = JSON.parse(event.data);
              // Only log when training starts, every 10 epochs, and when completed/failed
              if (data.status === 'running' && (data.current_epoch === 1 || data.current_epoch % 10 === 0)) {
                console.log(`🔄 Training Progress: Epoch ${data.current_epoch}/${data.config?.epochs || '?'} (${data.progress?.toFixed(1) || 0}%)`);
              }
              if (data.status === 'completed' || data.status === 'failed') {
                console.log('✅ Training completed or failed:', data.status);
              }
              setSessionData(data);
              if (data.status === 'completed' || data.status === 'failed') {
                setIsTraining(false);
                setMessage({
                  type: data.status === 'completed' ? 'success' : 'error',
                  text: data.status === 'completed'
                    ? `Training completed! Final accuracy: ${(data.final_accuracy * 100).toFixed(2)}%`
                    : `Training failed: ${data.error}`
                });
                websocket?.close();
                setActiveWebSocket(null);
              } else if (data.status === 'running') {
                setIsTraining(true);
              }
            } catch (error) {
              console.error('❌ Failed to parse WebSocket message:', error);
            }
          };
          
          websocket.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            setMessage({ type: 'error', text: 'WebSocket connection failed - please check backend is running' });
            setActiveWebSocket(null);
          };
          
          websocket.onclose = () => {
            // Only log on close
            console.log('WebSocket connection closed');
            setActiveWebSocket(null);
          };
          
        } catch (error) {
          console.error('❌ Failed to create WebSocket:', error);
          setMessage({ type: 'error', text: 'Failed to establish WebSocket connection - please check backend is running' });
        }
      } else {
        console.log('🔗 Active WebSocket already exists, skipping duplicate connection');
      }
    } else {
      console.log('📡 No active WebSocket found but also no currentSession');
    }

    return () => {
      if (websocket) {
        websocket.close();
      }
      // Don't close activeWebSocket here as it might be used elsewhere
    };
  }, [currentSession, activeWebSocket]);

  const startTraining = async () => {
    try {
      const response = await axios.post('/api/training/start', config);
      const sessionId = response.data.session_id;
      
      // Immediately set session and establish WebSocket connection
      setCurrentSession(sessionId);
      setIsTraining(true);
      setSessionData(null);
      setMessage({ type: 'info', text: 'Training started... Connecting to real-time updates...' });
      
      // Immediately connect WebSocket (don't wait for useEffect)
      console.log(`🚀 Training started with session: ${sessionId}`);
      const immediateWebSocket = connectWebSocket(sessionId);
      
      if (immediateWebSocket) {
        console.log('✅ Immediate WebSocket connection initiated');
      } else {
        console.log('⚠️ Immediate WebSocket failed, useEffect will handle fallback');
      }
      
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

      {/* Real-time Training Status */}
      {isTraining && sessionData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2" component="div">
            <strong>🔄 Real-time Training Update:</strong><br />
            Epoch {sessionData.current_epoch || 0}/{sessionData.config?.epochs || '?'} 
            ({sessionData.progress?.toFixed(1) || 0}% complete)
            {sessionData.current_metrics && (
              <>
                <br />
                <strong>📊 Current Metrics:</strong> Loss: {(sessionData.current_metrics.train_loss || 0).toFixed(4)}, 
                Acc: {((sessionData.current_metrics.train_accuracy || 0) * 100).toFixed(2)}%, 
                Val Loss: {(sessionData.current_metrics.val_loss || 0).toFixed(4)}, 
                Val Acc: {((sessionData.current_metrics.val_accuracy || 0) * 100).toFixed(2)}%
              </>
            )}
          </Typography>
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