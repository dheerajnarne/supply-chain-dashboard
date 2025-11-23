import React from 'react';
import { Box, CircularProgress } from '@mui/material';

const LoadingSpinner = ({ size = 40, color = 'primary', fullScreen = false }) => {
    if (fullScreen) {
        return (
            <Box
                sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    minHeight: '100vh',
                    width: '100%',
                }}
                className="animate-fade-in"
            >
                <CircularProgress size={size} color={color} />
            </Box>
        );
    }

    return (
        <Box
            sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                py: 4,
            }}
            className="animate-fade-in"
        >
            <CircularProgress size={size} color={color} />
        </Box>
    );
};

export default LoadingSpinner;
