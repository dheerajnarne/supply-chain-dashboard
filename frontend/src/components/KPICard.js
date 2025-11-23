import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';
import AnimatedCounter from './common/AnimatedCounter';

const KPICard = ({ title, value, trend, icon: Icon, delay = 0 }) => {
  const trendColor = trend >= 0 ? '#10b981' : '#ef4444';
  const TrendIcon = trend >= 0 ? TrendingUp : TrendingDown;

  // Extract numeric value
  const numericValue = typeof value === 'string'
    ? parseFloat(value.replace(/[^0-9.-]+/g, ''))
    : value;

  const prefix = typeof value === 'string' && value.includes('$') ? '$' : '';
  const suffix = typeof value === 'string' && value.includes('%') ? '%' : '';

  return (
    <Card
      className="card-hover animate-fade-in-up"
      style={{ animationDelay: `${delay}s` }}
      sx={{
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
        background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
        border: '1px solid #334155',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 4,
          background: 'linear-gradient(90deg, #0ea5e9 0%, #8b5cf6 100%)',
        },
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Typography
            variant="body2"
            color="text.secondary"
            fontWeight={500}
            textTransform="uppercase"
            letterSpacing="0.05em"
          >
            {title}
          </Typography>
          {Icon && (
            <Box
              sx={{
                width: 48,
                height: 48,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%)',
                borderRadius: 2,
                boxShadow: '0 4px 14px 0 rgba(14, 165, 233, 0.3)',
              }}
              className="animate-scale-in"
              style={{ animationDelay: `${delay + 0.2}s` }}
            >
              <Icon sx={{ color: 'white', fontSize: 24 }} />
            </Box>
          )}
        </Box>

        <Box sx={{ mb: 1.5 }}>
          {!isNaN(numericValue) ? (
            <AnimatedCounter
              end={numericValue}
              prefix={prefix}
              suffix={suffix}
            />
          ) : (
            <Typography variant="h4" component="div" fontWeight="bold">
              {value}
            </Typography>
          )}
        </Box>

        {trend !== undefined && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
            }}
          >
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                px: 1.5,
                py: 0.5,
                borderRadius: 1.5,
                bgcolor: trend >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              }}
            >
              <TrendIcon sx={{ fontSize: 16, color: trendColor }} />
              <Typography
                variant="body2"
                fontWeight={600}
                sx={{ color: trendColor }}
              >
                {Math.abs(trend)}%
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" fontSize="0.8125rem">
              vs last month
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default KPICard;
