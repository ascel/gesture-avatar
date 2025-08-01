import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  CameraAlt,
  Storage,
  School,
  PlayArrow,
  Analytics,
  CheckCircle,
  Warning,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

interface SystemStatus {
  gestures: Record<string, { name: string; sample_count: number }>;
  models: Array<{ name: string; is_active: boolean }>;
  total_samples: number;
}

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    gestures: {},
    models: [],
    total_samples: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSystemStatus();
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const [gesturesRes, modelsRes, edaRes] = await Promise.all([
        axios.get('/api/data/gestures'),
        axios.get('/api/models/list'),
        axios.get('/api/eda/dataset-info'),
      ]);

      setSystemStatus({
        gestures: gesturesRes.data.gestures,
        models: modelsRes.data.models,
        total_samples: edaRes.data.total_samples,
      });
    } catch (error) {
      console.error('Error fetching system status:', error);
    } finally {
      setLoading(false);
    }
  };

  const quickActions = [
    {
      title: 'Collect Data',
      description: 'Record new gesture samples',
      icon: <CameraAlt />,
      color: '#4caf50',
      action: () => navigate('/data-collection'),
    },
    {
      title: 'Train Model',
      description: 'Train new gesture recognition model',
      icon: <School />,
      color: '#ff9800',
      action: () => navigate('/training'),
    },
    {
      title: 'Live Demo',
      description: 'Test real-time gesture detection',
      icon: <PlayArrow />,
      color: '#2196f3',
      action: () => navigate('/live-demo'),
    },
    {
      title: 'Analytics',
      description: 'View data and model insights',
      icon: <Analytics />,
      color: '#9c27b0',
      action: () => navigate('/eda'),
    },
  ];

  const getSystemHealth = () => {
    const hasData = Object.keys(systemStatus.gestures).length > 0;
    const hasModels = systemStatus.models.length > 0;
    const hasActiveModel = systemStatus.models.some(m => m.is_active);

    if (hasData && hasModels && hasActiveModel) {
      return { status: 'healthy', message: 'System ready for use', color: '#4caf50' };
    } else if (hasData && hasModels) {
      return { status: 'warning', message: 'No active model selected', color: '#ff9800' };
    } else if (hasData) {
      return { status: 'warning', message: 'No trained models available', color: '#ff9800' };
    } else {
      return { status: 'error', message: 'No data collected yet', color: '#f44336' };
    }
  };

  const health = getSystemHealth();

  if (loading) {
    return (
      <Box sx={{ width: '100%' }}>
        <LinearProgress />
        <Typography sx={{ mt: 2, textAlign: 'center' }}>Loading dashboard...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      {/* System Health */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            {health.status === 'healthy' ? (
              <CheckCircle sx={{ color: health.color, mr: 1 }} />
            ) : (
              <Warning sx={{ color: health.color, mr: 1 }} />
            )}
            <Typography variant="h6">System Status</Typography>
            <Chip
              label={health.status.toUpperCase()}
              sx={{ ml: 'auto', backgroundColor: health.color, color: 'white' }}
            />
          </Box>
          <Typography variant="body1" color="textSecondary">
            {health.message}
          </Typography>
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Gesture Types
              </Typography>
              <Typography variant="h4">
                {Object.keys(systemStatus.gestures).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Samples
              </Typography>
              <Typography variant="h4">
                {systemStatus.total_samples}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Trained Models
              </Typography>
              <Typography variant="h4">
                {systemStatus.models.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Model
              </Typography>
              <Typography variant="h4">
                {systemStatus.models.some(m => m.is_active) ? '1' : '0'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Quick Actions */}
      <Typography variant="h5" gutterBottom>
        Quick Actions
      </Typography>
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {quickActions.map((action, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card
              sx={{
                cursor: 'pointer',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                },
              }}
              onClick={action.action}
            >
              <CardContent sx={{ textAlign: 'center' }}>
                <Box
                  sx={{
                    backgroundColor: action.color,
                    borderRadius: '50%',
                    width: 60,
                    height: 60,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 16px',
                    color: 'white',
                  }}
                >
                  {action.icon}
                </Box>
                <Typography variant="h6" gutterBottom>
                  {action.title}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {action.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Recent Data */}
      <Typography variant="h5" gutterBottom>
        Data Overview
      </Typography>
      <Grid container spacing={3}>
        {Object.entries(systemStatus.gestures).map(([gestureName, gestureData]) => (
          <Grid item xs={12} sm={6} md={4} key={gestureName}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {gestureData.name}
                </Typography>
                <Typography variant="h4" color="primary">
                  {gestureData.sample_count}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  samples collected
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Dashboard; 