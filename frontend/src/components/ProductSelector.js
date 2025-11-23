import React, { useState, useEffect } from 'react';
import { Autocomplete, TextField, CircularProgress, Box, Chip } from '@mui/material';
import apiService from '../api/api';

const ProductSelector = ({ onSelect, selectedProduct }) => {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchProducts = async () => {
      setLoading(true);
      try {
        const response = await apiService.getProducts(100);
        setProducts(response.data.products);
      } catch (error) {
        console.error('Error fetching products:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchProducts();
  }, []);

  return (
    <Autocomplete
      options={products}
      getOptionLabel={(option) => `${option.product_name} (ID: ${option.product_id})`}
      renderInput={(params) => (
        <TextField
          {...params}
          label="Select Product"
          placeholder="Search by name or ID"
          InputProps={{
            ...params.InputProps,
            endAdornment: (
              <>
                {loading && <CircularProgress color="inherit" size={20} />}
                {params.InputProps.endAdornment}
              </>
            ),
          }}
        />
      )}
      renderOption={(props, option) => (
        <Box component="li" {...props}>
          <Box>
            <Box fontWeight="600">{option.product_name}</Box>
            <Box display="flex" gap={1} mt={0.5}>
              <Chip label={option.category_name} size="small" color="primary" variant="outlined" />
              <Chip label={`${option.total_orders} orders`} size="small" />
            </Box>
          </Box>
        </Box>
      )}
      onChange={(event, value) => onSelect(value)}
      value={selectedProduct}
      loading={loading}
      isOptionEqualToValue={(option, value) => option.product_id === value?.product_id}
      sx={{ width: '100%' }}
    />
  );
};

export default ProductSelector;
