import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
} from '@mui/material';
import {
  Analytics,
  Refresh,
  Download,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import axios from 'axios';

interface DatasetInfo {
  raw_data: Record<string, number>;
  processed_data: Record<string, number>;
  total_samples: number;
}

const EDA: React.FC = () => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo>({
    raw_data: {},
    processed_data: {},
    total_samples: 0,
  });
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);

  useEffect(() => {
    fetchDatasetInfo();
  }, []);

  const fetchDatasetInfo = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/eda/dataset-info');
      setDatasetInfo(response.data);
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to fetch dataset information' });
    } finally {
      setLoading(false);
    }
  };

  const prepareBarChartData = () => {
    const gestures = Object.keys(datasetInfo.raw_data);
    return gestures.map(gesture => ({
      gesture: gesture.replace('_', ' ').toUpperCase(),
      raw: datasetInfo.raw_data[gesture] || 0,
      processed: datasetInfo.processed_data[gesture] || 0,
    }));
  };

  const preparePieChartData = () => {
    return Object.entries(datasetInfo.raw_data).map(([gesture, count]) => ({
      name: gesture.replace('_', ' ').toUpperCase(),
      value: count,
    }));
  };

  const barChartData = prepareBarChartData();
  const pieChartData = preparePieChartData();

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  const downloadReport = () => {
    const report = {
      dataset_summary: datasetInfo,
      generated_at: new Date().toISOString(),
      charts_data: {
        bar_chart: barChartData,
        pie_chart: pieChartData,
      }
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `eda_report_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    setMessage({ type: 'success', text: 'EDA report downloaded successfully' });
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Exploratory Data Analysis
        </Typography>
        <Typography>Loading dataset information...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Exploratory Data Analysis
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
                Total Samples
              </Typography>
              <Typography variant="h4">
                {datasetInfo.total_samples}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Gesture Types
              </Typography>
              <Typography variant="h4">
                {Object.keys(datasetInfo.raw_data).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Processed Samples
              </Typography>
              <Typography variant="h4">
                {Object.values(datasetInfo.processed_data).reduce((sum, count) => sum + count, 0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Avg. per Gesture
              </Typography>
              <Typography variant="h4">
                {Object.keys(datasetInfo.raw_data).length > 0 
                  ? Math.round(datasetInfo.total_samples / Object.keys(datasetInfo.raw_data).length)
                  : 0
                }
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Controls */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={fetchDatasetInfo}
        >
          Refresh Data
        </Button>
        <Button
          variant="contained"
          startIcon={<Download />}
          onClick={downloadReport}
          disabled={datasetInfo.total_samples === 0}
        >
          Download Report
        </Button>
      </Box>

      {datasetInfo.total_samples === 0 ? (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 4 }}>
            <Analytics sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="textSecondary" gutterBottom>
              No data available
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Collect some gesture data to see analytics here
            </Typography>
          </CardContent>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {/* Sample Distribution Bar Chart */}
          <Grid item xs={12} lg={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Sample Distribution by Gesture
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={barChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="gesture" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="raw" fill="#8884d8" name="Raw Samples" />
                    <Bar dataKey="processed" fill="#82ca9d" name="Processed Samples" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Gesture Distribution Pie Chart */}
          <Grid item xs={12} lg={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Gesture Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <PieChart>
                    <Pie
                      data={pieChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {pieChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Data Quality Insights */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Data Quality Insights
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Balance Score
                  </Typography>
                  {(() => {
                    const counts = Object.values(datasetInfo.raw_data);
                    const maxCount = Math.max(...counts);
                    const minCount = Math.min(...counts);
                    const balanceScore = counts.length > 0 ? (minCount / maxCount) * 100 : 0;
                    
                    return (
                      <Typography variant="body2" color={balanceScore > 80 ? 'success.main' : balanceScore > 60 ? 'warning.main' : 'error.main'}>
                        {balanceScore.toFixed(1)}% - {
                          balanceScore > 80 ? 'Well balanced' :
                          balanceScore > 60 ? 'Moderately balanced' :
                          'Imbalanced dataset'
                        }
                      </Typography>
                    );
                  })()}
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Sample Sufficiency
                  </Typography>
                  {(() => {
                    const avgSamples = datasetInfo.total_samples / Math.max(Object.keys(datasetInfo.raw_data).length, 1);
                    return (
                      <Typography variant="body2" color={avgSamples >= 100 ? 'success.main' : avgSamples >= 50 ? 'warning.main' : 'error.main'}>
                        {avgSamples.toFixed(0)} avg samples per gesture - {
                          avgSamples >= 100 ? 'Excellent' :
                          avgSamples >= 50 ? 'Good' :
                          'Need more data'
                        }
                      </Typography>
                    );
                  })()}
                </Box>

                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Processing Status
                  </Typography>
                  {(() => {
                    const rawTotal = Object.values(datasetInfo.raw_data).reduce((sum, count) => sum + count, 0);
                    const processedTotal = Object.values(datasetInfo.processed_data).reduce((sum, count) => sum + count, 0);
                    const processedRatio = rawTotal > 0 ? (processedTotal / rawTotal) * 100 : 0;
                    
                    return (
                      <Typography variant="body2" color={processedRatio > 90 ? 'success.main' : processedRatio > 50 ? 'warning.main' : 'error.main'}>
                        {processedRatio.toFixed(1)}% processed - {
                          processedRatio > 90 ? 'Ready for training' :
                          processedRatio > 50 ? 'Partially processed' :
                          'Needs preprocessing'
                        }
                      </Typography>
                    );
                  })()}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Recommendations */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recommendations
                </Typography>
                
                <Box sx={{ pl: 2 }}>
                  {(() => {
                    const recommendations = [];
                    const counts = Object.values(datasetInfo.raw_data);
                    const maxCount = Math.max(...counts);
                    const minCount = Math.min(...counts);
                    const avgSamples = datasetInfo.total_samples / Math.max(Object.keys(datasetInfo.raw_data).length, 1);
                    const balanceScore = counts.length > 0 ? (minCount / maxCount) * 100 : 0;
                    
                    if (avgSamples < 50) {
                      recommendations.push("Collect more samples for each gesture (aim for 50+ per gesture)");
                    }
                    
                    if (balanceScore < 60) {
                      recommendations.push("Balance your dataset by collecting more samples for underrepresented gestures");
                    }
                    
                    const rawTotal = Object.values(datasetInfo.raw_data).reduce((sum, count) => sum + count, 0);
                    const processedTotal = Object.values(datasetInfo.processed_data).reduce((sum, count) => sum + count, 0);
                    if (rawTotal > processedTotal) {
                      recommendations.push("Run preprocessing on your raw data before training");
                    }
                    
                    if (Object.keys(datasetInfo.raw_data).length < 5) {
                      recommendations.push("Consider adding more gesture types for a comprehensive model");
                    }
                    
                    if (recommendations.length === 0) {
                      recommendations.push("Your dataset looks good! You can proceed with model training.");
                    }
                    
                    return recommendations.map((rec, index) => (
                      <Typography key={index} variant="body2" paragraph>
                        • {rec}
                      </Typography>
                    ));
                  })()}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default EDA; 