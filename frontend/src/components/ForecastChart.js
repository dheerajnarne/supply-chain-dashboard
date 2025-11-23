import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from 'recharts';
import { format } from 'date-fns';

const ForecastChart = ({ data, showConfidence = true }) => {
  if (!data || data.length === 0) return null;

  const chartData = data.map((item) => ({
    date: format(new Date(item.date), 'MMM dd'),
    demand: item.demand,
    lower: item.lower_bound,
    upper: item.upper_bound,
    day: item.day,
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
        <XAxis dataKey="date" stroke="#666" style={{ fontSize: '12px' }} />
        <YAxis stroke="#666" style={{ fontSize: '12px' }} />
        <Tooltip
          contentStyle={{
            backgroundColor: '#fff',
            border: '1px solid #e0e0e0',
            borderRadius: '8px',
          }}
          formatter={(value) => value?.toFixed(1)}
        />
        <Legend wrapperStyle={{ paddingTop: '20px' }} />

        {showConfidence && (
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="#000000"
            fillOpacity={0.1}
            name="95% CI Upper"
          />
        )}
        {showConfidence && (
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="#000000"
            fillOpacity={0.1}
            name="95% CI Lower"
          />
        )}

        <Line
          type="monotone"
          dataKey="demand"
          stroke="#000000"
          strokeWidth={2}
          dot={{ fill: '#000000', r: 3 }}
          activeDot={{ r: 5, fill: '#000000' }}
          name="Forecasted Demand"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default ForecastChart;
