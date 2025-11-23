import React from 'react';
import { Box, Typography } from '@mui/material';
import { Inbox } from '@mui/icons-material';

const EmptyState = ({
    icon: Icon = Inbox,
    title = 'No Data Available',
    description = 'There is no data to display at the moment.',
    action
}) => {
    return (
        <Box
            className="animate-fade-in-up"
            sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                py: 8,
                px: 3,
                textAlign: 'center',
            }}
        >
            <Icon
                sx={{
                    fontSize: 80,
                    color: 'text.disabled',
                    mb: 2,
                    opacity: 0.5,
                }}
            />
            <Typography
                variant="h6"
                color="text.primary"
                gutterBottom
                fontWeight={600}
            >
                {title}
            </Typography>
            <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mb: 3, maxWidth: 400 }}
            >
                {description}
            </Typography>
            {action && <Box sx={{ mt: 2 }}>{action}</Box>}
        </Box>
    );
};

export default EmptyState;
