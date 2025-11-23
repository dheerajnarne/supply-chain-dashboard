import React from 'react';
import {
    Drawer,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Typography,
    Box,
    ListItemButton,
    Divider,
    Avatar,
} from '@mui/material';
import {
    ShowChart,
    Inventory,
    Warning,
    LocalShipping,
    Dashboard as DashboardIcon,
    Settings,
    Logout,
} from '@mui/icons-material';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';

const drawerWidth = 260;

const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Inventory', icon: <Inventory />, path: '/inventory' },
    { text: 'Anomalies', icon: <Warning />, path: '/anomalies' },
    { text: 'Routes', icon: <LocalShipping />, path: '/routes' },
];

const Sidebar = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { logout, user } = useAuth();

    const handleLogout = () => {
        logout();
        toast.success('Logged out successfully');
        navigate('/login');
    };

    return (
        <Drawer
            variant="permanent"
            className="animate-slide-in-left"
            sx={{
                width: drawerWidth,
                flexShrink: 0,
                '& .MuiDrawer-paper': {
                    width: drawerWidth,
                    boxSizing: 'border-box',
                    background: 'linear-gradient(180deg, #1e293b 0%, #0f172a 100%)',
                    borderRight: '1px solid #334155',
                },
            }}
        >
            {/* Logo Section */}
            <Box sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box
                    sx={{
                        width: 40,
                        height: 40,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: 'linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%)',
                        borderRadius: 2,
                    }}
                    className="animate-scale-in"
                >
                    <ShowChart fontSize="medium" sx={{ color: 'white' }} />
                </Box>
                <Typography
                    variant="h6"
                    fontWeight="800"
                    letterSpacing="-0.02em"
                    className="gradient-text"
                >
                    NEXUS AI
                </Typography>
            </Box>

            {/* User Info */}
            {user && (
                <Box
                    sx={{
                        px: 3,
                        pb: 2,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 2,
                        mb: 1,
                    }}
                    className="animate-fade-in"
                >
                    <Avatar
                        sx={{
                            width: 36,
                            height: 36,
                            bgcolor: 'primary.main',
                            fontSize: '0.875rem',
                            fontWeight: 600,
                        }}
                    >
                        {user.email?.[0]?.toUpperCase()}
                    </Avatar>
                    <Box sx={{ flex: 1, overflow: 'hidden' }}>
                        <Typography
                            variant="body2"
                            fontWeight={600}
                            color="text.primary"
                            sx={{
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                            }}
                        >
                            {user.full_name || 'User'}
                        </Typography>
                        <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                                display: 'block',
                            }}
                        >
                            {user.email}
                        </Typography>
                    </Box>
                </Box>
            )}

            <Divider sx={{ borderColor: '#334155', mb: 2 }} />

            {/* Navigation */}
            <List sx={{ px: 2, flex: 1 }}>
                {menuItems.map((item, index) => {
                    const active = location.pathname === item.path;
                    return (
                        <ListItem
                            key={item.text}
                            disablePadding
                            sx={{ mb: 0.5 }}
                            className="animate-fade-in"
                            style={{ animationDelay: `${index * 0.1}s` }}
                        >
                            <ListItemButton
                                onClick={() => navigate(item.path)}
                                className={active ? '' : 'ripple'}
                                sx={{
                                    borderRadius: 2,
                                    py: 1.5,
                                    px: 2.5,
                                    background: active
                                        ? 'linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%)'
                                        : 'transparent',
                                    color: active ? 'white' : 'text.secondary',
                                    position: 'relative',
                                    overflow: 'hidden',
                                    '&:hover': {
                                        background: active
                                            ? 'linear-gradient(135deg, #0284c7 0%, #7c3aed 100%)'
                                            : 'rgba(14, 165, 233, 0.08)',
                                        color: active ? 'white' : 'primary.main',
                                        transform: 'translateX(4px)',
                                    },
                                    '&::before': active ? {
                                        content: '""',
                                        position: 'absolute',
                                        left: 0,
                                        top: 0,
                                        bottom: 0,
                                        width: 4,
                                        background: 'white',
                                        borderRadius: '0 4px 4px 0',
                                    } : {},
                                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                                }}
                            >
                                <ListItemIcon
                                    sx={{
                                        minWidth: 40,
                                        color: 'inherit',
                                    }}
                                >
                                    {item.icon}
                                </ListItemIcon>
                                <ListItemText
                                    primary={item.text}
                                    primaryTypographyProps={{
                                        fontWeight: active ? 700 : 500,
                                        fontSize: '0.9375rem',
                                        letterSpacing: '0.01em',
                                    }}
                                />
                            </ListItemButton>
                        </ListItem>
                    );
                })}
            </List>

            <Divider sx={{ borderColor: '#334155', my: 2 }} />

            {/* Bottom Actions */}
            <List sx={{ px: 2, mb: 2 }}>
                <ListItem disablePadding sx={{ mb: 0.5 }}>
                    <ListItemButton
                        className="ripple"
                        sx={{
                            borderRadius: 2,
                            py: 1.5,
                            px: 2.5,
                            color: 'text.secondary',
                            '&:hover': {
                                color: 'primary.main',
                                bgcolor: 'rgba(14, 165, 233, 0.08)',
                                transform: 'translateX(4px)',
                            },
                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                        }}
                    >
                        <ListItemIcon sx={{ minWidth: 40, color: 'inherit' }}>
                            <Settings />
                        </ListItemIcon>
                        <ListItemText
                            primary="Settings"
                            primaryTypographyProps={{
                                fontWeight: 500,
                                fontSize: '0.9375rem',
                            }}
                        />
                    </ListItemButton>
                </ListItem>
                <ListItem disablePadding>
                    <ListItemButton
                        onClick={handleLogout}
                        className="ripple"
                        sx={{
                            borderRadius: 2,
                            py: 1.5,
                            px: 2.5,
                            color: 'text.secondary',
                            '&:hover': {
                                color: 'error.main',
                                bgcolor: 'rgba(239, 68, 68, 0.08)',
                                transform: 'translateX(4px)',
                            },
                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                        }}
                    >
                        <ListItemIcon sx={{ minWidth: 40, color: 'inherit' }}>
                            <Logout />
                        </ListItemIcon>
                        <ListItemText
                            primary="Logout"
                            primaryTypographyProps={{
                                fontWeight: 500,
                                fontSize: '0.9375rem',
                            }}
                        />
                    </ListItemButton>
                </ListItem>
            </List>
        </Drawer>
    );
};

export default Sidebar;
