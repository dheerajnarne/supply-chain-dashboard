import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { Warning, Error, Info } from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import apiService from '../api/api';

const AnomalyView = ({ productId, productName }) => {
  const [anomalyData, setAnomalyData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisDays, setAnalysisDays] = useState(90);

  useEffect(() => {
    const fetchAnomalies = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await apiService.detectAnomalies(productId, analysisDays);
        setAnomalyData(response.data);
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to detect anomalies');
      } finally {
        setLoading(false);
      }
    };

    if (productId) {
      fetchAnomalies();
    }
  }, [productId, analysisDays]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  if (!anomalyData) return null;

  const severityColors = {
    High: 'default',
    Medium: 'default',
    Low: 'default',
  };

  const severityIcons = {
    High: <Error sx={{ color: 'text.primary' }} />,
    Medium: <Warning sx={{ color: 'text.secondary' }} />,
    Low: <Info sx={{ color: 'text.disabled' }} />,
  };

  const chartData = Object.entries(anomalyData.severity_distribution).map(([key, value]) => ({
    name: key,
    count: value,
  }));

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" fontWeight="bold">
          {productName} - Anomaly Detection
        </Typography>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Analysis Period</InputLabel>
          <Select
            value={analysisDays}
            label="Analysis Period"
            onChange={(e) => setAnalysisDays(e.target.value)}
          >
            <MenuItem value={30}>30 Days</MenuItem>
            <MenuItem value={60}>60 Days</MenuItem>
            <MenuItem value={90}>90 Days</MenuItem>
            <MenuItem value={180}>180 Days</MenuItem>
            <MenuItem value={365}>365 Days</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" variant="subtitle2">
                Total Anomalies
              </Typography>
              <Typography variant="h3" fontWeight="bold" color="text.primary">
                {anomalyData.anomalies_detected}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                out of {anomalyData.analysis_period_days} days analyzed
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" variant="subtitle2">
                Anomaly Rate
              </Typography>
              <Typography variant="h3" fontWeight="bold" color="text.primary">
                {anomalyData.anomaly_rate.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                of total observations
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" variant="subtitle2">
                Severity Distribution
              </Typography>
              <Box display="flex" gap={1} mt={1}>
                {Object.entries(anomalyData.severity_distribution).map(([severity, count]) => (
                  <Chip
                    key={severity}
                    label={`${severity}: ${count}`}
                    color={severityColors[severity]}
                    size="small"
                  />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Severity Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Severity Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#000000" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Recommendations */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Recommendations
              </Typography>
              {anomalyData.recommendations.map((rec, index) => (
                <Alert
                  key={index}
                  severity="info"
                  variant="outlined"
                  sx={{ mb: 1, borderColor: 'text.secondary', color: 'text.primary' }}
                  icon={rec.includes('⚠️') || rec.includes('🔴') ? <Warning sx={{ color: 'text.primary' }} /> : <Info sx={{ color: 'text.secondary' }} />}
                >
                  {rec.replace(/[⚠️🔴✓]/g, '')}
                </Alert>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Top Anomalies Table */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Top Anomalies Detected
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Date</TableCell>
                      <TableCell>Severity</TableCell>
                      <TableCell align="right">Actual Demand</TableCell>
                      <TableCell align="right">Expected Demand</TableCell>
                      <TableCell align="right">Deviation</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {anomalyData.top_anomalies.map((anomaly, index) => (
                      <TableRow key={index} hover>
                        <TableCell>{anomaly.date}</TableCell>
                        <TableCell>
                          <Chip
                            icon={severityIcons[anomaly.severity]}
                            label={anomaly.severity}
                            color={severityColors[anomaly.severity]}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="right" sx={{ fontWeight: 600 }}>
                          {anomaly.demand.toFixed(1)}
                        </TableCell>
                        <TableCell align="right">{anomaly.expected.toFixed(1)}</TableCell>
                        <TableCell align="right" sx={{ color: 'text.primary', fontWeight: 600 }}>
                          {((anomaly.demand - anomaly.expected) / anomaly.expected * 100).toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AnomalyView;
