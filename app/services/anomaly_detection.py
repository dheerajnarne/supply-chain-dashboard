
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from fastapi.concurrency import run_in_threadpool
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    
    def __init__(self):
        self.contamination = 0.05 # Expected percentage of anomalies
        
    def _detect_anomalies_sync(self, df: pd.DataFrame, days: int) -> dict:
        # Filter for the analysis period
        cutoff_date = df['order_date'].max() - pd.Timedelta(days=days)
        df_filtered = df[df['order_date'] >= cutoff_date].copy()
        
        if len(df_filtered) < 10:
            raise ValueError("Insufficient data for anomaly detection")
            
        # Aggregate daily demand
        daily_data = df_filtered.groupby('order_date').agg({
            'order_quantity': 'sum',
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        # Prepare features for Isolation Forest
        features = ['order_quantity', 'sales', 'profit']
        X = daily_data[features].values
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        anomalies = iso_forest.fit_predict(X_scaled)
        
        daily_data['is_anomaly'] = anomalies == -1
        
        # Calculate stats
        total_days = len(daily_data)
        anomaly_count = daily_data['is_anomaly'].sum()
        anomaly_rate = (anomaly_count / total_days) * 100
        
        # Identify top anomalies
        anomalies_df = daily_data[daily_data['is_anomaly']].copy()
        
        # Calculate expected values (simple moving average as baseline)
        daily_data['expected_demand'] = daily_data['order_quantity'].rolling(window=7, min_periods=1, center=True).mean()
        
        top_anomalies = []
        severity_distribution = {'High': 0, 'Medium': 0, 'Low': 0}
        
        for _, row in anomalies_df.iterrows():
            # Determine severity based on deviation from mean
            deviation = abs(row['order_quantity'] - daily_data['order_quantity'].mean()) / daily_data['order_quantity'].std()
            
            if deviation > 3:
                severity = 'High'
            elif deviation > 2:
                severity = 'Medium'
            else:
                severity = 'Low'
                
            severity_distribution[severity] += 1
            
            # Get expected demand from rolling average, fallback to mean
            expected = daily_data.loc[daily_data['order_date'] == row['order_date'], 'expected_demand'].values[0]
            if np.isnan(expected):
                expected = daily_data['order_quantity'].mean()
            
            top_anomalies.append({
                'date': row['order_date'].strftime('%Y-%m-%d'),
                'severity': severity,
                'demand': float(row['order_quantity']),
                'expected': float(expected),
                'deviation': float(row['order_quantity'] - expected)
            })
            
        # Sort by deviation magnitude
        top_anomalies.sort(key=lambda x: abs(x['deviation']), reverse=True)
        
        # Generate recommendations
        recommendations = []
        if severity_distribution['High'] > 0:
            recommendations.append("🔴 Critical anomalies detected. Immediate investigation required.")
        if anomaly_rate > 10:
            recommendations.append("⚠️ High anomaly rate detected. Process stability concerns.")
        else:
            recommendations.append("✓ Anomaly rate within acceptable limits.")
            
        if severity_distribution['Medium'] > severity_distribution['High']:
             recommendations.append("⚠️ Monitor medium severity anomalies for potential patterns.")

        return {
            'anomalies_detected': int(anomaly_count),
            'analysis_period_days': days,
            'anomaly_rate': float(anomaly_rate),
            'severity_distribution': severity_distribution,
            'top_anomalies': top_anomalies[:10],
            'recommendations': recommendations
        }

    async def detect_anomalies(self, df: pd.DataFrame, days: int = 90) -> dict:
        return await run_in_threadpool(self._detect_anomalies_sync, df, days)
