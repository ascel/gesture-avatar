import React, { useState, useRef, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  Alert,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  CameraAlt,
  Stop,
  Save,
  Refresh,
  Transform,
} from '@mui/icons-material';
import Webcam from 'react-webcam';
import axios from 'axios';

interface GestureSession {
  session_id: string;
  gesture_name: string;
  samples_collected: number;
  is_active: boolean;
}

const DataCollection: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const [currentSession, setCurrentSession] = useState<GestureSession | null>(null);
  const [gestureName, setGestureName] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [samplesCollected, setSamplesCollected] = useState(0);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [availableGestures, setAvailableGestures] = useState<Record<string, any>>({});

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: 'user',
  };

  React.useEffect(() => {
    fetchAvailableGestures();
  }, []);

  const fetchAvailableGestures = async () => {
    try {
      const response = await axios.get('/api/data/gestures');
      setAvailableGestures(response.data.gestures);
    } catch (error) {
      console.error('Error fetching gestures:', error);
    }
  };

  const startSession = async () => {
    if (!gestureName.trim()) {
      setMessage({ type: 'error', text: 'Please enter a gesture name' });
      return;
    }

    try {
      const response = await axios.post(`/api/data/start-collection?gesture_name=${gestureName}`);
      setCurrentSession({
        session_id: response.data.session_id,
        gesture_name: gestureName,
        samples_collected: 0,
        is_active: true,
      });
      setSamplesCollected(0);
      setMessage({ type: 'success', text: `Started collecting data for "${gestureName}"` });
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to start collection session' });
    }
  };

  const captureFrame = useCallback(async () => {
    if (!webcamRef.current || !currentSession) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      const response = await axios.post('/api/data/save-sample', {
        gesture_name: currentSession.gesture_name,
        timestamp: new Date().toISOString(),
        frame_data: imageSrc,
      });

      const newCount = samplesCollected + 1;
      setSamplesCollected(newCount);
      
      if (response.data.landmarks_count) {
        setMessage({ 
          type: 'success', 
          text: `Captured sample ${newCount} with ${response.data.landmarks_count} landmarks` 
        });
      } else {
        setMessage({ type: 'success', text: `Captured sample ${newCount}` });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to save sample' });
    }
  }, [currentSession, samplesCollected]);

  const startRecording = () => {
    setIsRecording(true);
    const interval = setInterval(() => {
      captureFrame();
    }, 500); // Capture every 500ms

    // Stop after 10 seconds
    setTimeout(() => {
      clearInterval(interval);
      setIsRecording(false);
    }, 10000);
  };

  const endSession = () => {
    setCurrentSession(null);
    setIsRecording(false);
    fetchAvailableGestures();
    setMessage({ type: 'info', text: 'Collection session ended' });
  };

  const convertExistingData = async () => {
    try {
      setMessage({ type: 'info', text: 'Converting existing data to landmarks...' });
      
      const response = await axios.post('/api/data/convert-existing');
      
      if (response.data.error) {
        setMessage({ type: 'error', text: response.data.error });
      } else {
        setMessage({ 
          type: 'success', 
          text: response.data.message 
        });
        fetchAvailableGestures(); // Refresh the gesture list
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to convert existing data' });
    }
  };

  const predefinedGestures = ['fist', 'open_hand', 'peace', 'point', 'thumbs_up'];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Collection
      </Typography>

      {message && (
        <Alert severity={message.type} sx={{ mb: 2 }} onClose={() => setMessage(null)}>
          {message.text}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Webcam Feed */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Camera Feed
              </Typography>
              <Box sx={{ position: 'relative', display: 'inline-block' }}>
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  height={480}
                  width={640}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  style={{ borderRadius: '8px' }}
                />
                {isRecording && (
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 10,
                      left: 10,
                      backgroundColor: 'red',
                      color: 'white',
                      px: 2,
                      py: 1,
                      borderRadius: 1,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                    }}
                  >
                    <Box
                      sx={{
                        width: 8,
                        height: 8,
                        backgroundColor: 'white',
                        borderRadius: '50%',
                        animation: 'blink 1s infinite',
                      }}
                    />
                    RECORDING
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Controls */}
        <Grid item xs={12} md={4}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Collection Controls
              </Typography>

              {!currentSession ? (
                <Box>
                  <TextField
                    fullWidth
                    label="Gesture Name"
                    value={gestureName}
                    onChange={(e) => setGestureName(e.target.value)}
                    sx={{ mb: 2 }}
                    placeholder="e.g., wave, thumbs_up"
                  />

                  <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                    Quick select:
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    {predefinedGestures.map((gesture) => (
                      <Chip
                        key={gesture}
                        label={gesture}
                        onClick={() => setGestureName(gesture)}
                        sx={{ mr: 1, mb: 1 }}
                        variant={gestureName === gesture ? 'filled' : 'outlined'}
                      />
                    ))}
                  </Box>

                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<CameraAlt />}
                    onClick={startSession}
                    disabled={!gestureName.trim()}
                  >
                    Start Collection
                  </Button>
                </Box>
              ) : (
                <Box>
                  <Typography variant="body1" sx={{ mb: 2 }}>
                    Collecting: <strong>{currentSession.gesture_name}</strong>
                  </Typography>
                  <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                    Samples collected: {samplesCollected}
                  </Typography>

                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={isRecording ? <Stop /> : <CameraAlt />}
                    onClick={isRecording ? () => setIsRecording(false) : startRecording}
                    disabled={isRecording}
                    sx={{ mb: 1 }}
                    color={isRecording ? 'error' : 'primary'}
                  >
                    {isRecording ? 'Recording...' : 'Start Recording (10s)'}
                  </Button>

                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Save />}
                    onClick={captureFrame}
                    disabled={isRecording}
                    sx={{ mb: 1 }}
                  >
                    Capture Single Frame
                  </Button>

                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Stop />}
                    onClick={endSession}
                    color="error"
                  >
                    End Session
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Current Data Overview */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Data
              </Typography>
              {Object.keys(availableGestures).length === 0 ? (
                <Typography variant="body2" color="textSecondary">
                  No data collected yet
                </Typography>
              ) : (
                Object.entries(availableGestures).map(([name, data]: [string, any]) => (
                  <Box key={name} sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">{name}</Typography>
                      <Chip label={`${data.sample_count} samples`} size="small" />
                    </Box>
                  </Box>
                ))
              )}
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Refresh />}
                onClick={fetchAvailableGestures}
                sx={{ mt: 1 }}
                size="small"
              >
                Refresh
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Transform />}
                onClick={convertExistingData}
                sx={{ mt: 1 }}
                size="small"
                color="secondary"
              >
                Convert Existing Data
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </Box>
  );
};

export default DataCollection; 