import React from 'react';
import { Box, Chip } from '@mui/material';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import { motion } from 'framer-motion';

const RealtimeIndicator = ({ status }) => {
  const statusConfig = {
    connected: { label: 'Live', color: 'success', bgColor: 'rgba(46, 125, 50, 0.1)' },
    disconnected: { label: 'Offline', color: 'default', bgColor: 'rgba(158, 158, 158, 0.1)' },
    error: { label: 'Error', color: 'error', bgColor: 'rgba(211, 47, 47, 0.1)' },
  };

  const config = statusConfig[status] || statusConfig.disconnected;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Chip
        icon={
          <motion.div
            animate={{
              scale: status === 'connected' ? [1, 1.2, 1] : 1,
            }}
            transition={{
              duration: 2,
              repeat: status === 'connected' ? Infinity : 0,
            }}
          >
            <FiberManualRecordIcon sx={{ fontSize: 12 }} />
          </motion.div>
        }
        label={config.label}
        color={config.color}
        size="small"
        sx={{
          bgcolor: config.bgColor,
          backdropFilter: 'blur(10px)',
          fontWeight: 600,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      />
    </motion.div>
  );
};

export default RealtimeIndicator;
