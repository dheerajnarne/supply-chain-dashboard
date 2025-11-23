import React from 'react';
import { Box, Card, CardContent } from '@mui/material';

const SkeletonCard = ({ height = 140 }) => {
    return (
        <Card className="animate-pulse-soft">
            <CardContent>
                <Box
                    sx={{
                        height: 24,
                        width: '40%',
                        bgcolor: 'rgba(100, 116, 139, 0.3)',
                        borderRadius: 1,
                        mb: 2,
                    }}
                    className="shimmer"
                />
                <Box
                    sx={{
                        height: 40,
                        width: '60%',
                        bgcolor: 'rgba(100, 116, 139, 0.3)',
                        borderRadius: 1,
                        mb: 1,
                    }}
                    className="shimmer"
                />
                <Box
                    sx={{
                        height: 20,
                        width: '30%',
                        bgcolor: 'rgba(100, 116, 139, 0.3)',
                        borderRadius: 1,
                    }}
                    className="shimmer"
                />
            </CardContent>
        </Card>
    );
};

export default SkeletonCard;
