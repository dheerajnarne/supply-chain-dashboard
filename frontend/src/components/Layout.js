import React from 'react';
import { Box, CssBaseline } from '@mui/material';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './Sidebar';
import Dashboard from './Dashboard';

const Layout = () => {
    return (
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            <CssBaseline />
            <Sidebar />
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    p: 4,
                    width: { sm: `calc(100% - 260px)` },
                    minHeight: '100vh',
                    position: 'relative',
                }}
            >
                <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/inventory" element={<Dashboard />} />
                    <Route path="/anomalies" element={<Dashboard />} />
                    <Route path="/routes" element={<Dashboard />} />
                </Routes>
            </Box>
        </Box>
    );
};

export default Layout;
