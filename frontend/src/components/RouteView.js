import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Button,
  TextField,
  Chip,
} from '@mui/material';
import { Search, LocalShipping } from '@mui/icons-material';
import apiService from '../api/api';

const RouteView = () => {
  const [routeData, setRouteData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedDate, setSelectedDate] = useState('2017-10-01');

  const handleOptimize = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.optimizeRoutes(selectedDate, 5);
      setRouteData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to optimize routes');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Route Optimization
      </Typography>

      <Box display="flex" gap={2} mb={3}>
        <TextField
          type="date"
          label="Select Date"
          value={selectedDate}
          onChange={(e) => setSelectedDate(e.target.value)}
          InputLabelProps={{ shrink: true }}
        />
        <Button
          variant="contained"
          startIcon={<Search />}
          onClick={handleOptimize}
          disabled={loading}
        >
          Optimize Routes
        </Button>
      </Box>

      {loading && (
        <Box display="flex" justifyContent="center" py={4}>
          <CircularProgress />
        </Box>
      )}

      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

      {routeData && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" variant="subtitle2">
                  Total Orders
                </Typography>
                <Typography variant="h3" fontWeight="bold">
                  {routeData.total_orders}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" variant="subtitle2">
                  Routes Created
                </Typography>
                <Typography variant="h3" fontWeight="bold">
                  {routeData.num_routes}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" variant="subtitle2">
                  Est. Savings
                </Typography>
                <Typography variant="h3" fontWeight="bold" color="text.primary">
                  {routeData.estimated_savings.toFixed(1)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {routeData.routes.map((route) => (
            <Grid item xs={12} md={6} key={route.route_id}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <LocalShipping sx={{ color: 'text.primary' }} />
                    <Typography variant="h6" fontWeight="bold">
                      Route {route.route_id + 1}
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between" mb={2}>
                    <Typography color="text.secondary">Orders:</Typography>
                    <Typography fontWeight="600">{route.num_orders}</Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between" mb={2}>
                    <Typography color="text.secondary">Total Demand:</Typography>
                    <Typography fontWeight="600">{route.total_demand.toFixed(0)} units</Typography>
                  </Box>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Cities:
                  </Typography>
                  <Box display="flex" gap={0.5} flexWrap="wrap">
                    {route.cities.map((city, idx) => (
                      <Chip key={idx} label={city} size="small" />
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
};

export default RouteView;
