import React, { useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    TextField,
    Button,
    Link,
    Alert,
    CircularProgress,
    Container
} from '@mui/material';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { ShowChart } from '@mui/icons-material';

const LoginPage = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await login(email, password);
            navigate('/');
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to login');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container maxWidth="sm">
            <Box
                display="flex"
                flexDirection="column"
                alignItems="center"
                justifyContent="center"
                minHeight="100vh"
            >
                <Paper
                    elevation={0}
                    sx={{
                        p: 5,
                        width: '100%',
                        border: '1px solid #e5e7eb',
                        borderRadius: 0,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                    }}
                >
                    <Box
                        sx={{
                            width: 48,
                            height: 48,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'common.black',
                            border: '2px solid black',
                            mb: 3,
                        }}
                    >
                        <ShowChart />
                    </Box>

                    <Typography variant="h4" fontWeight="800" gutterBottom letterSpacing="-0.02em">
                        Welcome Back
                    </Typography>
                    <Typography color="text.secondary" mb={4}>
                        Sign in to access your dashboard
                    </Typography>

                    {error && (
                        <Alert severity="error" sx={{ width: '100%', mb: 3 }} variant="outlined">
                            {error}
                        </Alert>
                    )}

                    <form onSubmit={handleSubmit} style={{ width: '100%' }}>
                        <TextField
                            fullWidth
                            label="Email Address"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            margin="normal"
                            required
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            margin="normal"
                            required
                            sx={{ mb: 3 }}
                        />
                        <Button
                            fullWidth
                            size="large"
                            type="submit"
                            variant="contained"
                            disabled={loading}
                            sx={{ mb: 3, height: 48 }}
                        >
                            {loading ? <CircularProgress size={24} color="inherit" /> : 'Sign In'}
                        </Button>

                        <Box textAlign="center">
                            <Typography variant="body2" color="text.secondary">
                                Don't have an account?{' '}
                                <Link component={RouterLink} to="/signup" color="primary" fontWeight="600">
                                    Sign up
                                </Link>
                            </Typography>
                        </Box>
                    </form>
                </Paper>
            </Box>
        </Container>
    );
};

export default LoginPage;
