import React from 'react';
import { Box, Skeleton, Grid, Card, CardContent } from '@mui/material';

export const KPISkeleton = () => (
  <Grid container spacing={3}>
    {[1, 2, 3, 4].map((i) => (
      <Grid item xs={12} sm={6} md={3} key={i}>
        <Card>
          <CardContent>
            <Skeleton variant="text" width="60%" height={30} />
            <Skeleton variant="text" width="80%" height={60} />
          </CardContent>
        </Card>
      </Grid>
    ))}
  </Grid>
);

export const ChartSkeleton = () => (
  <Card>
    <CardContent>
      <Skeleton variant="text" width="40%" height={40} sx={{ mb: 2 }} />
      <Skeleton variant="rectangular" width="100%" height={400} sx={{ borderRadius: 2 }} />
    </CardContent>
  </Card>
);
