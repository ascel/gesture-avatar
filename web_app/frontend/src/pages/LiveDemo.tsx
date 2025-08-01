import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Alert,
  Chip,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Paper
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Videocam,
  VideocamOff,
  Settings,
  Refresh
} from '@mui/icons-material';
import Webcam from 'react-webcam';
import axios from 'axios';

interface GestureResult {
  gesture: string;
  confidence: number;
  avatar_frame: string;
  additional_info: any;
}

interface ModelInfo {
  name: string;
  path: string;
  is_active: boolean;
  accuracy?: number;
}

const LiveDemo: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const isRunningRef = useRef<boolean>(false);
  
  const [isRunning, setIsRunning] = useState(false);
  const [isWebcamReady, setIsWebcamReady] = useState(false);
  const [currentGesture, setCurrentGesture] = useState<string>('unknown');
  const [confidence, setConfidence] = useState<number>(0);
  const [avatarFrame, setAvatarFrame] = useState<string>('');
  const [fps, setFps] = useState<number>(0);
  const [error, setError] = useState<string>('');
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeModel, setActiveModel] = useState<string>('');
  const [showWebcam, setShowWebcam] = useState(true);
  const [showAvatar, setShowAvatar] = useState(true);
  const [detectionThreshold, setDetectionThreshold] = useState(0.5);
  
  // Performance tracking
  const [frameCount, setFrameCount] = useState(0);
  const [lastFpsTime, setLastFpsTime] = useState(Date.now());
  
  // Gesture history for smoothing
  const [gestureHistory, setGestureHistory] = useState<string[]>([]);
  const maxHistoryLength = 5;

  // Load available models on component mount
  useEffect(() => {
    loadModels();
  }, []);

  // Cleanup interval on component unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        console.log('Cleaning up interval on unmount');
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);

  const loadModels = async () => {
    try {
      const response = await axios.get('/api/models/list');
      setModels(response.data.models);
      
      // Find active model or best model
      let active = response.data.models.find((model: ModelInfo) => model.is_active);
      
      // If no active model, try to activate the best model
      if (!active && response.data.models.length > 0) {
        const bestModel = response.data.models.find((model: ModelInfo) => 
          model.name.includes('best') || model.name.includes('feature')
        );
        if (bestModel) {
          try {
            await activateModel(bestModel.path);
            active = bestModel;
          } catch (error) {
            console.log('Could not auto-activate best model:', error);
          }
        }
      }
      
      if (active) {
        setActiveModel(active.path);
      }
    } catch (error) {
      console.error('Error loading models:', error);
      setError('Failed to load models. Please ensure at least one model is trained.');
    }
  };

  const activateModel = async (modelPath: string) => {
    try {
      await axios.post('/api/models/activate', modelPath);
      setActiveModel(modelPath);
      setError('');
      
      // Reload models to update active status
      loadModels();
    } catch (error) {
      console.error('Error activating model:', error);
      setError('Failed to activate model');
    }
  };

  const captureFrame = useCallback(async () => {
    console.log('captureFrame called, isRunning:', isRunningRef.current, 'webcamRef:', !!webcamRef.current); // Debug log
    if (!webcamRef.current || !isRunningRef.current) return;

    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;

      // Send frame to backend for gesture detection
      console.log('Sending frame to backend...'); // Debug log
      const response = await axios.post('/api/inference/detect', {
        frame_data: imageSrc
      });
      console.log('Received response from backend:', response.status); // Debug log

      const result: GestureResult = response.data;
      
      console.log('Gesture detection result:', result); // Debug log
      
      // Update gesture history for smoothing
      setGestureHistory(prev => {
        const newHistory = [...prev, result.gesture].slice(-maxHistoryLength);
        
        // Use most common gesture in recent history if confidence is high enough
        if (result.confidence >= detectionThreshold) {
          const gestureCount: { [key: string]: number } = {};
          newHistory.forEach(g => {
            gestureCount[g] = (gestureCount[g] || 0) + 1;
          });
          
          const mostCommon = Object.entries(gestureCount)
            .sort(([,a], [,b]) => b - a)[0][0];
          
          setCurrentGesture(mostCommon);
        } else {
          // If confidence is low, keep current gesture or set to unknown
          if (result.confidence < 0.3) {
            setCurrentGesture('unknown');
          }
        }
        
        return newHistory;
      });
      
      setConfidence(result.confidence);
      setAvatarFrame(result.avatar_frame);
      
      // Update FPS
      setFrameCount(prev => prev + 1);
      const now = Date.now();
      if (now - lastFpsTime >= 1000) {
        setFps(frameCount);
        setFrameCount(0);
        setLastFpsTime(now);
      }
      
      setError('');
    } catch (error) {
      console.error('Error detecting gesture:', error);
      setError('Gesture detection failed - check console for details');
      // Set fallback values
      setCurrentGesture('unknown');
      setConfidence(0);
    }
  }, [detectionThreshold]);

  const startDemo = async () => {
    // Allow demo to start even without active model (will use simulated results)
    isRunningRef.current = true;
    setIsRunning(true);
    setError('');
    setFrameCount(0);
    setLastFpsTime(Date.now());
    
    // Clear any existing interval first
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    // Start capture loop
    console.log('Starting capture loop...'); // Debug log
    intervalRef.current = setInterval(captureFrame, 100); // 10 FPS
    console.log('Interval set with ID:', intervalRef.current); // Debug log
  };

  const stopDemo = () => {
    console.log('Stopping demo, clearing interval...'); // Debug log
    isRunningRef.current = false;
    setIsRunning(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      console.log('Interval cleared'); // Debug log
    }
    setCurrentGesture('unknown');
    setConfidence(0);
    setFps(0);
  };

  const resetDemo = () => {
    stopDemo();
    setGestureHistory([]);
    setAvatarFrame('');
    setError('');
  };

  const getGestureDisplayName = (gesture: string): string => {
    const gestureMap: { [key: string]: string } = {
      'gesture_0': 'Fist',
      'gesture_1': 'Open Hand',
      'gesture_2': 'Peace',
      'gesture_3': 'Point',
      'gesture_4': 'Thumbs Up',
      'fist': 'Fist',
      'open_hand': 'Open Hand',
      'peace': 'Peace',
      'point': 'Point',
      'thumbs_up': 'Thumbs Up',
      'no_hand': 'No Hand',
      'unknown': 'Unknown'
    };
    return gestureMap[gesture] || gesture;
  };

  const getConfidenceColor = (conf: number): "success" | "warning" | "error" => {
    if (conf >= 0.8) return "success";
    if (conf >= 0.5) return "warning";
    return "error";
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Live Gesture Demo
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Real-time gesture detection and avatar animation
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Control Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Controls
              </Typography>
              
              {/* Model Selection */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Active Model</InputLabel>
                <Select
                  value={activeModel}
                  label="Active Model"
                  onChange={(e) => activateModel(e.target.value)}
                  disabled={isRunning}
                >
                  {models.map((model) => (
                    <MenuItem key={model.path} value={model.path}>
                      {model.name} {model.accuracy && `(${(model.accuracy * 100).toFixed(1)}%)`}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Detection Threshold */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Confidence Threshold</InputLabel>
                <Select
                  value={detectionThreshold}
                  label="Confidence Threshold"
                  onChange={(e) => setDetectionThreshold(Number(e.target.value))}
                  disabled={isRunning}
                >
                  <MenuItem value={0.3}>Low (0.3)</MenuItem>
                  <MenuItem value={0.5}>Medium (0.5)</MenuItem>
                  <MenuItem value={0.7}>High (0.7)</MenuItem>
                  <MenuItem value={0.9}>Very High (0.9)</MenuItem>
                </Select>
              </FormControl>

              {/* Display Options */}
              <FormControlLabel
                control={
                  <Switch
                    checked={showWebcam}
                    onChange={(e) => setShowWebcam(e.target.checked)}
                  />
                }
                label="Show Webcam"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={showAvatar}
                    onChange={(e) => setShowAvatar(e.target.checked)}
                  />
                }
                label="Show Avatar"
              />

              {/* Action Buttons */}
              <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {!isRunning ? (
                  <Button
                    variant="contained"
                    startIcon={<PlayArrow />}
                    onClick={startDemo}
                    disabled={!isWebcamReady}
                  >
                    Start Demo
                  </Button>
                ) : (
                  <Button
                    variant="contained"
                    color="error"
                    startIcon={<Stop />}
                    onClick={stopDemo}
                  >
                    Stop Demo
                  </Button>
                )}
                
                <Button
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={resetDemo}
                  disabled={isRunning}
                >
                  Reset
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={<Settings />}
                  onClick={loadModels}
                  disabled={isRunning}
                >
                  Refresh Models
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* Status Panel */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Status
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Current Gesture
                </Typography>
                <Chip 
                  label={getGestureDisplayName(currentGesture)}
                  color={currentGesture !== 'unknown' && currentGesture !== 'no_hand' ? 'primary' : 'default'}
                  size="small"
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Confidence: {(confidence * 100).toFixed(1)}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={confidence * 100}
                  color={getConfidenceColor(confidence)}
                />
              </Box>

              <Typography variant="body2" color="text.secondary">
                FPS: {fps}
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                Status: {isRunning ? 'Running' : 'Stopped'}
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                Model: {activeModel ? 'Active' : 'None Selected'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Video Display */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Live Feed
              </Typography>
              
              <Grid container spacing={2}>
                {/* Webcam Feed */}
                {showWebcam && (
                  <Grid item xs={12} md={showAvatar ? 6 : 12}>
                    <Paper sx={{ p: 1, textAlign: 'center' }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Webcam
                      </Typography>
                      <Webcam
                        ref={webcamRef}
                        audio={false}
                        screenshotFormat="image/jpeg"
                        width="100%"
                        height="auto"
                        onUserMedia={() => setIsWebcamReady(true)}
                        onUserMediaError={() => {
                          setIsWebcamReady(false);
                          setError('Failed to access webcam');
                        }}
                        style={{ maxWidth: '400px', borderRadius: '8px' }}
                      />
                    </Paper>
                  </Grid>
                )}

                {/* Avatar Display */}
                {showAvatar && (
                  <Grid item xs={12} md={showWebcam ? 6 : 12}>
                    <Paper sx={{ p: 1, textAlign: 'center' }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Avatar
                      </Typography>
                      <Box
                        sx={{
                          width: '100%',
                          maxWidth: '400px',
                          height: '300px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          backgroundColor: 'grey.100',
                          borderRadius: '8px',
                          mx: 'auto'
                        }}
                      >
                        {avatarFrame ? (
                          <img
                            src={avatarFrame}
                            alt="Avatar"
                            style={{
                              maxWidth: '100%',
                              maxHeight: '100%',
                              borderRadius: '8px'
                            }}
                          />
                        ) : (
                          <Typography color="text.secondary">
                            {isRunning ? 'Waiting for gesture...' : 'Start demo to see avatar'}
                          </Typography>
                        )}
                      </Box>
                    </Paper>
                  </Grid>
                )}
              </Grid>

              {!isWebcamReady && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <VideocamOff />
                    Please allow webcam access to start the demo
                  </Box>
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default LiveDemo; 