import React, { useEffect, useState } from 'react';
import { Typography } from '@mui/material';

const AnimatedCounter = ({ end, duration = 2000, suffix = '', prefix = '' }) => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        let startTime;
        let animationFrame;

        const animate = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const progress = timestamp - startTime;
            const percentage = Math.min(progress / duration, 1);

            // Easing function for smooth animation
            const easeOutQuart = 1 - Math.pow(1 - percentage, 4);
            const current = Math.floor(end * easeOutQuart);

            setCount(current);

            if (progress < duration) {
                animationFrame = requestAnimationFrame(animate);
            } else {
                setCount(end);
            }
        };

        animationFrame = requestAnimationFrame(animate);

        return () => {
            if (animationFrame) {
                cancelAnimationFrame(animationFrame);
            }
        };
    }, [end, duration]);

    return (
        <Typography variant="h4" component="span" fontWeight="bold">
            {prefix}{count.toLocaleString()}{suffix}
        </Typography>
    );
};

export default AnimatedCounter;
