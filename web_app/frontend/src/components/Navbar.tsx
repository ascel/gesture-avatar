import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Dashboard as DashboardIcon,
  CameraAlt as CameraIcon,
  Settings as SettingsIcon,
  School as TrainingIcon,
  Storage as ModelsIcon,
  PlayArrow as LiveIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';

const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    { path: '/dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/data-collection', label: 'Data Collection', icon: <CameraIcon /> },
    { path: '/preprocessing', label: 'Preprocessing', icon: <SettingsIcon /> },
    { path: '/training', label: 'Training', icon: <TrainingIcon /> },
    { path: '/models', label: 'Models', icon: <ModelsIcon /> },
    { path: '/live-demo', label: 'Live Demo', icon: <LiveIcon /> },
    { path: '/eda', label: 'Analytics', icon: <AnalyticsIcon /> },
  ];

  return (
    <AppBar position="static" sx={{ backgroundColor: '#1e1e1e' }}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Gesture Avatar Platform
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          {menuItems.map((item) => (
            <Button
              key={item.path}
              color="inherit"
              startIcon={item.icon}
              onClick={() => navigate(item.path)}
              sx={{
                backgroundColor: location.pathname === item.path ? 'rgba(25, 118, 210, 0.2)' : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(25, 118, 210, 0.1)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 