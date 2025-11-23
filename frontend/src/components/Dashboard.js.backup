import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  Alert,
} from '@mui/material';
import {
  ShowChart,
  Inventory,
  Warning,
  LocalShipping,
  Refresh,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';

import ProductSelector from './ProductSelector';
import ForecastView from './ForecastView';
import InventoryView from './InventoryView';
import AnomalyView from './AnomalyView';
import RouteView from './RouteView';
import KPICard from './KPICard';
import RealtimeIndicator from './RealtimeIndicator';
import SkeletonCard from './common/SkeletonCard';
import EmptyState from './common/EmptyState';
import apiService from '../api/api';
import { useWebSocket } from '../hooks/useWebSocket';

const Dashboard = () => {
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [dashboardMetrics, setDashboardMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const location = useLocation();
  const navigate = useNavigate();

  const getTabValue = () => {
    switch (location.pathname) {
      case '/inventory': return 1;
      case '/anomalies': return 2;
      case '/routes': return 3;
      default: return 0;
    }
  };

  const activeTab = getTabValue();

  const { data: wsData, status: wsStatus } = useWebSocket(
    'ws://localhost:8000/api/v1/ws',
    true
  );

  const fetchDashboardMetrics = async () => {
    try {
      setLoading(true);
      const response = await apiService.getDashboardMetrics();
      setDashboardMetrics(response.data);
      setError(null);
    } catch (err) {
      console.error('Dashboard metrics error:', err);
      setError('Failed to load dashboard metrics');
      toast.error('Failed to load dashboard metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardMetrics();
  }, []);

  useEffect(() => {
    if (wsData?.type === 'metrics_updated') {
      setDashboardMetrics(wsData.data);
    }
  }, [wsData]);

  const handleProductSelect = (product) => {
    setSelectedProduct(product);
    if (activeTab !== 0 && activeTab !== 1 && activeTab !== 2) {
      navigate('/');
    }
  };

  return (
    <Box className="animate-fade-in">
      <Container maxWidth="xl" disableGutters>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
          <Box>
            <Typography variant="h4" color="text.primary" gutterBottom fontWeight={700}>
              Overview
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Welcome back, here's what's happening with your supply chain today.
            </Typography>
          </Box>
          <Box display="flex" gap={2} alignItems="center">
            <RealtimeIndicator status={wsStatus} />
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={fetchDashboardMetrics}
              className="ripple"
              sx={{
                borderColor: 'primary.main',
                color: 'primary.main',
                '&:hover': {
                  borderColor: 'primary.dark',
                  bgcolor: 'rgba(14, 165, 233, 0.08)',
                },
              }}
            >
              Refresh
            </Button>
          </Box>
        </Box>

        {/* KPI Cards */}
        <Grid container spacing={3} mb={4}>
          {loading ? (
            <>
              <Grid item xs={12} sm={6} md={3}>
                <SkeletonCard />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <SkeletonCard />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <SkeletonCard />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <SkeletonCard />
              </Grid>
            </>
          ) : dashboardMetrics ? (
            <>
              <Grid item xs={12} sm={6} md={3}>
                <KPICard
                  title="Total Products"
                  value={dashboardMetrics.total_products}
                  icon={Inventory}
                  delay={0}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <KPICard
                  title="Total Orders"
                  value={dashboardMetrics.total_orders?.toLocaleString()}
                  icon={ShowChart}
                  trend={8.5}
                  delay={0.1}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <KPICard
                  title="Total Sales"
                  value={`$${(dashboardMetrics.total_sales / 1000000).toFixed(2)}M`}
                  icon={LocalShipping}
                  trend={12.3}
                  delay={0.2}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <KPICard
                  title="Avg Profit"
                  value={`$${dashboardMetrics.avg_profit?.toFixed(2)}`}
                  icon={Warning}
                  trend={-2.4}
                  delay={0.3}
                />
              </Grid>
            </>
          ) : null}
        </Grid>

        {/* Product Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Paper sx={{ p: 3, mb: 3 }}>
            <ProductSelector
              onSelect={handleProductSelect}
              selectedProduct={selectedProduct}
            />
          </Paper>
        </motion.div>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Main Content */}
        <Paper sx={{ overflow: 'hidden', minHeight: 600 }}>
          <Box p={3}>
            <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ mb: 3 }}>
              {activeTab === 0 && "Demand Forecast"}
              {activeTab === 1 && "Inventory Optimization"}
              {activeTab === 2 && "Anomaly Detection"}
              {activeTab === 3 && "Route Optimization"}
            </Typography>

            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                {activeTab === 3 ? (
                  <RouteView />
                ) : selectedProduct ? (
                  <>
                    {activeTab === 0 && (
                      <ForecastView
                        productId={selectedProduct.product_id}
                        productName={selectedProduct.product_name}
                      />
                    )}
                    {activeTab === 1 && (
                      <InventoryView
                        productId={selectedProduct.product_id}
                        productName={selectedProduct.product_name}
                      />
                    )}
                    {activeTab === 2 && (
                      <AnomalyView
                        productId={selectedProduct.product_id}
                        productName={selectedProduct.product_name}
                      />
                    )}
                  </>
                ) : (
                  <EmptyState
                    icon={ShowChart}
                    title="Select a Product"
                    description="Choose a product from the selector above to view detailed analytics."
                  />
                )}
              </motion.div>
            </AnimatePresence>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
};

export default Dashboard;
