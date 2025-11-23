import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Download,
  TrendingUp,
  TrendingDown,
  Remove,
  CalendarToday,
} from '@mui/icons-material';
import { format } from 'date-fns';
import ForecastChart from './ForecastChart';
import apiService from '../api/api';
import { useWebSocket } from '../hooks/useWebSocket';

const ForecastView = ({ productId, productName }) => {
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [forecastDays, setForecastDays] = useState(30);
  const [showCI, setShowCI] = useState(true);
  const [realtimeEnabled, setRealtimeEnabled] = useState(false);

  // WebSocket for real-time forecast updates
  const { data: wsData, status: wsStatus } = useWebSocket(
    `ws://localhost:8000/ws/forecast/${productId}`,
    realtimeEnabled
  );

  const fetchForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.createForecast(productId, forecastDays, showCI);
      setForecastData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate forecast');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (productId) {
      fetchForecast();
    }
  }, [productId, forecastDays, showCI]);

  // Update from WebSocket
  useEffect(() => {
    if (wsData?.type === 'forecast_update' && realtimeEnabled) {
      setForecastData(wsData.data);
    }
  }, [wsData, realtimeEnabled]);

  const getTrendIcon = (direction) => {
    if (direction === 'increasing') return <TrendingUp sx={{ color: 'text.primary' }} />;
    if (direction === 'decreasing') return <TrendingDown sx={{ color: 'text.secondary' }} />;
    return <Remove color="disabled" />;
  };

  const downloadCSV = () => {
    if (!forecastData) return;

    const csv = [
      ['Date', 'Day', 'Demand', 'Lower Bound', 'Upper Bound'],
      ...forecastData.daily_forecast.map((item) => [
        item.date,
        item.day,
        item.demand.toFixed(2),
        item.lower_bound?.toFixed(2) || '',
        item.upper_bound?.toFixed(2) || '',
      ]),
    ]
      .map((row) => row.join(','))
      .join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forecast_${productName}_${format(new Date(), 'yyyy-MM-dd')}.csv`;
    a.click();
  };

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

  if (!forecastData) return null;

  const { validation_metrics, eda_insights, daily_forecast } = forecastData;

  return (
    <Box>
      {/* Header Controls */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h5" fontWeight="bold" gutterBottom>
            {productName} - Demand Forecast
          </Typography>
          <Box display="flex" gap={1} alignItems="center">
            <Chip
              icon={<CalendarToday />}
              label={`${forecastData.forecast_start_date} to ${forecastData.forecast_end_date}`}
              variant="outlined"
              sx={{ borderRadius: 0 }}
            />
            <Chip
              label={`Accuracy: ${validation_metrics.accuracy.toFixed(1)}%`}
              variant="outlined"
              sx={{ borderRadius: 0, borderColor: 'text.primary', fontWeight: 600 }}
            />
            {realtimeEnabled && (
              <Chip
                label="Live Updates"
                size="small"
                variant="outlined"
                sx={{ animation: 'pulse 2s infinite', borderRadius: 0 }}
              />
            )}
          </Box>
        </Box>

        <Box display="flex" gap={2} alignItems="center">
          <FormControlLabel
            control={
              <Switch
                checked={realtimeEnabled}
                onChange={(e) => setRealtimeEnabled(e.target.checked)}
              />
            }
            label="Real-time"
          />
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Forecast Days</InputLabel>
            <Select
              value={forecastDays}
              label="Forecast Days"
              onChange={(e) => setForecastDays(e.target.value)}
            >
              <MenuItem value={7}>7 Days</MenuItem>
              <MenuItem value={14}>14 Days</MenuItem>
              <MenuItem value={21}>21 Days</MenuItem>
              <MenuItem value={30}>30 Days</MenuItem>
              <MenuItem value={60}>60 Days</MenuItem>
              <MenuItem value={90}>90 Days</MenuItem>
            </Select>
          </FormControl>
          <FormControlLabel
            control={<Switch checked={showCI} onChange={(e) => setShowCI(e.target.checked)} />}
            label="Show CI"
          />
          <Button variant="outlined" startIcon={<Download />} onClick={downloadCSV}>
            Export
          </Button>
        </Box>
      </Box>

      {/* Metrics Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" variant="subtitle2">
                Total Demand
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {forecastData.total_demand.toFixed(0)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                units
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" variant="subtitle2">
                Avg Daily Demand
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {forecastData.avg_daily_demand.toFixed(1)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                units/day
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" variant="subtitle2">
                Forecast Accuracy
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {validation_metrics.accuracy.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                MAPE: {validation_metrics.mape.toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="text.secondary" variant="subtitle2">
                    Trend
                  </Typography>
                  <Typography variant="h6" fontWeight="bold" sx={{ textTransform: 'capitalize' }}>
                    {eda_insights.trend_direction}
                  </Typography>
                </Box>
                {getTrendIcon(eda_insights.trend_direction)}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Forecast Chart */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            Demand Forecast Visualization
          </Typography>
          <ForecastChart data={daily_forecast} showConfidence={showCI} />
        </CardContent>
      </Card>

      {/* Insights Grid */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                EDA Insights
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">Mean Demand</Typography>
                  <Typography fontWeight="600">{eda_insights.mean.toFixed(1)} units</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">Weekend Avg</Typography>
                  <Typography fontWeight="600">
                    {eda_insights.weekend_avg.toFixed(1)} units
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">Weekday Avg</Typography>
                  <Typography fontWeight="600">
                    {eda_insights.weekday_avg.toFixed(1)} units
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Model Performance
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">MAE</Typography>
                  <Typography fontWeight="600">{validation_metrics.mae.toFixed(2)}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">RMSE</Typography>
                  <Typography fontWeight="600">{validation_metrics.rmse.toFixed(2)}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">MAPE</Typography>
                  <Typography fontWeight="600">{validation_metrics.mape.toFixed(2)}%</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Detailed Forecast Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            Daily Forecast Details
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Date</TableCell>
                  <TableCell>Day</TableCell>
                  <TableCell align="right">Demand</TableCell>
                  {showCI && <TableCell align="right">Lower Bound</TableCell>}
                  {showCI && <TableCell align="right">Upper Bound</TableCell>}
                </TableRow>
              </TableHead>
              <TableBody>
                {daily_forecast.map((item, index) => (
                  <TableRow key={index} hover>
                    <TableCell>{item.date}</TableCell>
                    <TableCell>
                      <Chip
                        label={item.day}
                        size="small"
                        color={['Saturday', 'Sunday'].includes(item.day) ? 'error' : 'default'}
                      />
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: 600 }}>
                      {item.demand.toFixed(1)}
                    </TableCell>
                    {showCI && <TableCell align="right">{item.lower_bound?.toFixed(1)}</TableCell>}
                    {showCI && <TableCell align="right">{item.upper_bound?.toFixed(1)}</TableCell>}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ForecastView;
