import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from '@mui/material';
import {
  Storage,
  CheckCircle,
  Refresh,
  Download,
  Delete,
} from '@mui/icons-material';
import axios from 'axios';

interface Model {
  name: string;
  path: string;
  size: number;
  modified: string;
  is_active: boolean;
  accuracy?: number;
  model_type?: string;
  timestamp?: string;
  training_date?: string;
  final_accuracy?: number;
}

const ModelManagement: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<Model | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/models/list');
      const ms: Model[] = response.data.models || [];
      setModels(ms);
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to fetch models' });
    } finally {
      setLoading(false);
    }
  };

  const activateModel = async (modelPath: string) => {
    try {
      await axios.post('/api/models/activate', null, {
        params: { model_path: modelPath }
      });
      setMessage({ type: 'success', text: 'Model activated successfully' });
      fetchModels(); // Refresh the list
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to activate model' });
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getModelTypeColor = (type?: string) => {
    switch (type) {
      case 'resnet':
        return 'primary';
      case 'efficientnet':
        return 'secondary';
      case 'efficientnet1d':
        return 'warning';
      default:
        return 'default';
    }
  };

  const requestDeleteModel = (model: Model) => {
    setPendingDelete(model);
    setConfirmOpen(true);
  };

  const handleCloseConfirm = () => {
    setConfirmOpen(false);
    setPendingDelete(null);
  };

  const deleteModel = async () => {
    if (!pendingDelete) return;
    try {
      await axios.delete(`/api/models/${encodeURIComponent(pendingDelete.name)}`, {
        params: { model_path: pendingDelete.path }
      });
      setMessage({ type: 'success', text: `Deleted model: ${pendingDelete.name}` });
      handleCloseConfirm();
      fetchModels();
    } catch (error) {
      setMessage({ type: 'error', text: `Failed to delete model: ${pendingDelete.name}` });
    }
  };

  // No sorting controls; show final accuracy only

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Management
      </Typography>

      {message && (
        <Alert severity={message.type} sx={{ mb: 2 }} onClose={() => setMessage(null)}>
          {message.text}
        </Alert>
      )}

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Models
              </Typography>
              <Typography variant="h4">
                {models.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Models
              </Typography>
              <Typography variant="h4">
                {models.filter(m => m.is_active).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                ResNet Models
              </Typography>
              <Typography variant="h4">
                {models.filter(m => m.model_type === 'resnet').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                EfficientNet1D Models
              </Typography>
              <Typography variant="h4">
                {models.filter(m => m.model_type === 'efficientnet1d').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Models Table */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Available Models
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={fetchModels}
                disabled={loading}
              >
                Refresh
              </Button>
            </Box>
          </Box>

          {loading ? (
            <Typography>Loading models...</Typography>
          ) : models.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Storage sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="textSecondary" gutterBottom>
                No models found
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Train your first model to see it here
              </Typography>
            </Box>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Model Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Accuracy</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Modified</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {models.map((model) => (
                    <TableRow key={model.name}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="medium">
                          {model.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={model.model_type || 'Unknown'}
                          size="small"
                          color={getModelTypeColor(model.model_type)}
                        />
                      </TableCell>
                      <TableCell>
                        {typeof model.final_accuracy === 'number' ? (
                          <Chip size="small" color="success" label={`${(model.final_accuracy * 100).toFixed(2)}%`} />
                        ) : (
                          <Typography variant="body2" color="textSecondary">N/A</Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatFileSize(model.size)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="textSecondary">
                          {formatDate(model.modified)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {model.is_active ? (
                          <Chip
                            icon={<CheckCircle />}
                            label="Active"
                            size="small"
                            color="success"
                          />
                        ) : (
                          <Chip
                            label="Inactive"
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          {!model.is_active && (
                            <Button
                              size="small"
                              variant="contained"
                              onClick={() => activateModel(model.path)}
                            >
                              Activate
                            </Button>
                          )}
                          <IconButton size="small" color="primary">
                            <Download />
                          </IconButton>
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => requestDeleteModel(model)}
                            aria-label={`Delete ${model.name}`}
                          >
                            <Delete />
                          </IconButton>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Delete confirmation dialog */}
      <Dialog open={confirmOpen} onClose={handleCloseConfirm}>
        <DialogTitle>Delete Model</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {pendingDelete
              ? `Are you sure you want to delete model "${pendingDelete.name}"? This action cannot be undone.`
              : 'Are you sure you want to delete this model?'}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseConfirm}>Cancel</Button>
          <Button onClick={deleteModel} color="error" variant="contained">Delete</Button>
        </DialogActions>
      </Dialog>

      {/* Information Card */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Model Management Guide
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            This page allows you to manage your trained gesture recognition models:
          </Typography>
          
          <Box sx={{ pl: 2 }}>
            <Typography variant="body2" paragraph>
              <strong>Activate Model:</strong> Set a model as active for real-time inference. Only one model can be active at a time.
            </Typography>
            
            <Typography variant="body2" paragraph>
              <strong>Model Types:</strong> ResNet and EfficientNet models are supported, each with different performance characteristics.
            </Typography>
            
            <Typography variant="body2" paragraph>
              <strong>Performance:</strong> Check the accuracy column to compare model performance before activation.
            </Typography>
            
            <Typography variant="body2">
              <strong>Storage:</strong> Models are stored locally and can be downloaded for backup or sharing.
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ModelManagement; 