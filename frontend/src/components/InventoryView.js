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
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Inventory2,
  Warning,
  CheckCircle,
  ShoppingCart,
  TrendingUp,
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import apiService from '../api/api';

const InventoryView = ({ productId, productName }) => {
  const [inventoryData, setInventoryData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchInventory = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await apiService.optimizeInventory(productId);
        setInventoryData(response.data);
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to optimize inventory');
      } finally {
        setLoading(false);
      }
    };

    if (productId) {
      fetchInventory();
    }
  }, [productId]);

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

  if (!inventoryData) return null;

  const riskColors = {
    Low: 'success',
    Medium: 'warning',
    High: 'error',
  };

  const chartData = [
    { name: 'Reorder Point', value: inventoryData.reorder_point, color: '#000000' },
    { name: 'Safety Stock', value: inventoryData.safety_stock, color: '#757575' },
    { name: 'EOQ', value: inventoryData.optimal_order_quantity, color: '#bdbdbd' },
  ];

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        {productName} - Inventory Optimization
      </Typography>

      {/* Risk Alert */}
      <Alert
        severity="info"
        variant="outlined"
        icon={<Warning sx={{ color: 'text.primary' }} />}
        sx={{ mb: 3, borderColor: 'text.primary', color: 'text.primary' }}
      >
        <Typography fontWeight="600">
          Stockout Risk: {inventoryData.stockout_risk}
        </Typography>
        <Typography variant="body2">
          Current stock covers approximately {inventoryData.current_stock_days} days of demand
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {/* Key Metrics */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <Inventory2 sx={{ color: 'text.primary' }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Reorder Point
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight="bold">
                {inventoryData.reorder_point.toFixed(0)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                units (95% service level)
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <Warning sx={{ color: 'text.primary' }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Safety Stock
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight="bold">
                {inventoryData.safety_stock.toFixed(0)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                units (buffer inventory)
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <ShoppingCart sx={{ color: 'text.primary' }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Optimal Order Qty
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight="bold">
                {inventoryData.optimal_order_quantity.toFixed(0)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                units (EOQ)
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Visualization */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Inventory Breakdown
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry) => `${entry.name}: ${entry.value.toFixed(0)}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
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
              <List>
                {inventoryData.recommendations.map((rec, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemIcon>
                        {rec.includes('⚠️') ? (
                          <Warning sx={{ color: 'text.secondary' }} />
                        ) : (
                          <CheckCircle sx={{ color: 'text.primary' }} />
                        )}
                      </ListItemIcon>
                      <ListItemText primary={rec.replace(/[⚠️✓]/g, '')} />
                    </ListItem>
                    {index < inventoryData.recommendations.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
              <Box mt={2}>
                <Chip
                  label={`Model Confidence: ${(inventoryData.model_confidence * 100).toFixed(0)}%`}
                  variant="outlined"
                  icon={<TrendingUp />}
                  sx={{ borderRadius: 0 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default InventoryView;
