import React, { useState, useRef, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  Chip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Checkbox,
} from '@mui/material';
import {
  CameraAlt,
  Stop,
  Save,
  Refresh,
  Transform,
  DeleteOutline,
  CleaningServices,
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
  const [cleanupOpen, setCleanupOpen] = useState(false);
  const [cleanupOptions, setCleanupOptions] = useState({ deleteRaw: true, deleteProcessed: true });
  const [confirmDelete, setConfirmDelete] = useState<{ open: boolean; gesture?: string }>({ open: false });

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

      setSamplesCollected((prev) => prev + 1);
    } catch (error) {
      if (!isRecording) {
        setMessage({ type: 'error', text: 'Failed to save sample' });
      }
    }
  }, [currentSession, isRecording]);

  const [recordDurationMs, setRecordDurationMs] = useState(10000);
  const [recordIntervalMs, setRecordIntervalMs] = useState(500);
  const [recordingProgress, setRecordingProgress] = useState(0);
  const recordTimerRef = React.useRef<number | null>(null);
  const recordTickerRef = React.useRef<number | null>(null);

  const stopRecording = () => {
    setIsRecording(false);
    setRecordingProgress(0);
    if (recordTickerRef.current) {
      window.clearInterval(recordTickerRef.current);
      recordTickerRef.current = null;
    }
    if (recordTimerRef.current) {
      window.clearTimeout(recordTimerRef.current);
      recordTimerRef.current = null;
    }
  };

  const startRecording = () => {
    if (isRecording) return;
    setIsRecording(true);
    setRecordingProgress(0);

    // Schedule frame captures
    const ticker = window.setInterval(() => {
      captureFrame();
    }, Math.max(100, recordIntervalMs));
    recordTickerRef.current = ticker as unknown as number;

    const startTs = Date.now();
    // Progress updater
    const progressTimer = window.setInterval(() => {
      const elapsed = Date.now() - startTs;
      const pct = Math.min(100, Math.round((elapsed / Math.max(1000, recordDurationMs)) * 100));
      setRecordingProgress(pct);
    }, 100);

    // Hard stop timer
    const stopper = window.setTimeout(() => {
      if (recordTickerRef.current) window.clearInterval(recordTickerRef.current);
      if (progressTimer) window.clearInterval(progressTimer);
      stopRecording();
    }, Math.max(1000, recordDurationMs));

    recordTimerRef.current = stopper as unknown as number;
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

  const clearGesture = (name: string) => {
    setConfirmDelete({ open: true, gesture: name });
  };

  const handleConfirmDelete = async () => {
    if (!confirmDelete.gesture) return;
    try {
      const response = await axios.delete(`/api/data/gestures/${encodeURIComponent(confirmDelete.gesture)}`);
      setMessage({ type: 'success', text: response.data.message || `Cleared data for ${confirmDelete.gesture}` });
      setConfirmDelete({ open: false, gesture: undefined });
      fetchAvailableGestures();
    } catch (error) {
      setMessage({ type: 'error', text: `Failed to clear data for ${confirmDelete.gesture}` });
      setConfirmDelete({ open: false, gesture: undefined });
    }
  };

  const openCleanup = () => setCleanupOpen(true);
  const closeCleanup = () => setCleanupOpen(false);
  const confirmCleanup = async () => {
    if (!cleanupOptions.deleteRaw && !cleanupOptions.deleteProcessed) {
      setMessage({ type: 'error', text: 'Select at least one option to clean' });
      return;
    }
    try {
      const response = await axios.post('/api/data/cleanup', {
        delete_raw: cleanupOptions.deleteRaw,
        delete_processed: cleanupOptions.deleteProcessed,
      });
      setMessage({ type: 'success', text: 'Cleanup completed' });
      setCleanupOpen(false);
      // Reset local state
      setCurrentSession(null);
      setIsRecording(false);
      setSamplesCollected(0);
      fetchAvailableGestures();
    } catch (error) {
      setMessage({ type: 'error', text: 'Cleanup failed' });
    }
  };

  const predefinedGestures = ['fist', 'open_hand', 'peace', 'point', 'thumbs_up'];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Collection
      </Typography>

      <Dialog open={Boolean(message)} onClose={() => setMessage(null)} maxWidth="xs" fullWidth>
        <DialogTitle>
          {message?.type === 'success' ? 'Success' : message?.type === 'error' ? 'Error' : 'Info'}
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2">{message?.text}</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMessage(null)}>OK</Button>
        </DialogActions>
      </Dialog>

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
                <Box
                  sx={{
                    position: 'absolute',
                    top: 10,
                    right: 10,
                    backgroundColor: 'rgba(0,0,0,0.6)',
                    color: 'white',
                    px: 1.5,
                    py: 0.5,
                    borderRadius: 1,
                    fontWeight: 600,
                  }}
                >
                  {samplesCollected}
                </Box>
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
                    onClick={isRecording ? stopRecording : startRecording}
                    sx={{ mb: 1 }}
                    color={isRecording ? 'error' : 'primary'}
                  >
                    {isRecording ? 'Stop Recording' : `Start Recording (${Math.round(recordDurationMs/1000)}s)`}
                  </Button>

                  {isRecording && (
                    <Box sx={{ mb: 2 }}>
                      <LinearProgress variant="determinate" value={recordingProgress} />
                      <Typography variant="caption" color="textSecondary">
                        {recordingProgress}%
                      </Typography>
                    </Box>
                  )}

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

                  <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                    <TextField
                      label="Duration (s)"
                      type="number"
                      size="small"
                      value={Math.round(recordDurationMs/1000)}
                      onChange={(e) => setRecordDurationMs(Math.max(1, Number(e.target.value)) * 1000)}
                      sx={{ width: 140 }}
                      inputProps={{ min: 1 }}
                    />
                    <TextField
                      label="Interval (ms)"
                      type="number"
                      size="small"
                      value={recordIntervalMs}
                      onChange={(e) => setRecordIntervalMs(Math.max(100, Number(e.target.value)))}
                      sx={{ width: 160 }}
                      inputProps={{ min: 100, step: 50 }}
                    />
                  </Box>

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
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2">{name}</Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip label={`${data.sample_count} samples`} size="small" />
                        <Button
                          size="small"
                          variant="text"
                          color="error"
                          startIcon={<DeleteOutline />}
                          onClick={() => clearGesture(name)}
                        >
                          Clear
                        </Button>
                      </Box>
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

              <Button
                fullWidth
                variant="contained"
                startIcon={<CleaningServices />}
                onClick={openCleanup}
                sx={{ mt: 1 }}
                size="small"
                color="warning"
              >
                Flush All Data
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Dialog open={cleanupOpen} onClose={closeCleanup} maxWidth="xs" fullWidth>
        <DialogTitle>Flush Collected Data</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 1 }}>
            This will permanently delete selected data.
          </Typography>
          <FormControlLabel
            control={
              <Checkbox
                checked={cleanupOptions.deleteRaw}
                onChange={(e) => setCleanupOptions((o) => ({ ...o, deleteRaw: e.target.checked }))}
              />
            }
            label="Delete raw collected samples"
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={cleanupOptions.deleteProcessed}
                onChange={(e) => setCleanupOptions((o) => ({ ...o, deleteProcessed: e.target.checked }))}
              />
            }
            label="Delete processed datasets and artifacts"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={closeCleanup}>Cancel</Button>
          <Button onClick={confirmCleanup} color="warning" variant="contained">Flush</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={confirmDelete.open} onClose={() => setConfirmDelete({ open: false, gesture: undefined })} maxWidth="xs" fullWidth>
        <DialogTitle>Delete Gesture Data</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            Delete all samples for gesture "{confirmDelete.gesture}"? This cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDelete({ open: false, gesture: undefined })}>Cancel</Button>
          <Button onClick={handleConfirmDelete} color="error" variant="contained">Delete</Button>
        </DialogActions>
      </Dialog>

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