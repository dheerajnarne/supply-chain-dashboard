
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from app.core.config import settings
from fastapi.concurrency import run_in_threadpool
import logging

logger = logging.getLogger(__name__)

class RouteOptimizer:
    
    def __init__(self):
        self.n_clusters = settings.KMEANS_N_CLUSTERS
        self.model = None
        self.scaler = StandardScaler()
        
    def _prepare_location_data_sync(self, df: pd.DataFrame) -> pd.DataFrame:
        customer_locations = df.groupby('customer_id').agg({
            'latitude': 'first',
            'longitude': 'first',
            'customer_city': 'first',
            'customer_state': 'first',
            'order_id': 'count',
            'sales': 'sum',
            'days_for_shipping_real': 'mean'
        }).reset_index()
        
        customer_locations.columns = [
            'customer_id', 'latitude', 'longitude', 'customer_city',
            'customer_state', 'total_orders', 'total_sales', 'avg_delivery_days'
        ]
        
        customer_locations = customer_locations.dropna(subset=['latitude', 'longitude'])
        customer_locations = customer_locations[(customer_locations['latitude'] != 0) & 
                                                 (customer_locations['longitude'] != 0)]
        
        return customer_locations
    
    def _optimize_routes_sync(self, location_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(location_data) < self.n_clusters:
            logger.warning(f"Not enough unique locations for {self.n_clusters} clusters")
            return location_data, pd.DataFrame()
        
        coordinates = location_data[['latitude', 'longitude']].values
        coordinates_scaled = self.scaler.fit_transform(coordinates)
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=settings.KMEANS_MAX_ITER,
            random_state=42,
            n_init=10
        )
        
        location_data['cluster_id'] = self.model.fit_predict(coordinates_scaled)
        
        cluster_centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        
        location_data['cluster_center_lat'] = location_data['cluster_id'].map(
            lambda x: cluster_centers[x][0]
        )
        location_data['cluster_center_lon'] = location_data['cluster_id'].map(
            lambda x: cluster_centers[x][1]
        )
        
        location_data['distance_to_center'] = np.sqrt(
            (location_data['latitude'] - location_data['cluster_center_lat'])**2 +
            (location_data['longitude'] - location_data['cluster_center_lon'])**2
        ) * 111
        
        cluster_summary = location_data.groupby('cluster_id').agg({
            'customer_id': 'count',
            'total_orders': 'sum',
            'total_sales': 'sum',
            'avg_delivery_days': 'mean',
            'distance_to_center': 'mean',
            'cluster_center_lat': 'first',
            'cluster_center_lon': 'first',
            'customer_city': lambda x: list(set(x))[:5] # Top 5 cities
        }).reset_index()
        
        cluster_summary.columns = [
            'cluster_id', 'customer_count', 'total_orders', 'total_sales',
            'avg_delivery_days', 'avg_distance_to_center',
            'cluster_center_lat', 'cluster_center_lon', 'cities'
        ]
        
        return location_data, cluster_summary

    async def optimize_routes(self, df: pd.DataFrame) -> dict:
        location_data = await run_in_threadpool(self._prepare_location_data_sync, df)
        _, cluster_summary = await run_in_threadpool(self._optimize_routes_sync, location_data)
        
        # Calculate metrics for frontend
        total_orders = int(cluster_summary['total_orders'].sum()) if not cluster_summary.empty else 0
        num_routes = len(cluster_summary)
        
        # Mock estimated savings calculation (in reality would compare to baseline)
        estimated_savings = 15.5  # Placeholder percentage
        
        routes = []
        for _, row in cluster_summary.iterrows():
            routes.append({
                'route_id': int(row['cluster_id']),
                'num_orders': int(row['total_orders']),
                'total_demand': float(row['total_orders']), 
                'cities': row['cities'] 
            })
            
        # To get cities, we need to join back or aggregate differently. 
        # Let's improve _optimize_routes_sync to return cities list.
        
        return {
            'total_orders': total_orders,
            'num_routes': num_routes,
            'estimated_savings': estimated_savings,
            'routes': routes
        }
