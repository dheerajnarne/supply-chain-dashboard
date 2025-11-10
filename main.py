# main.py - Complete Backend Application
# Supply Chain Predictive Analytics Dashboard

from fastapi import FastAPI, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index, select, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func as sql_func
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, AsyncGenerator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import asyncio
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/supply_chain_db"
    
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Supply Chain Analytics Dashboard"
    DEBUG: bool = True
    
    KAGGLE_DATASET_PATH: str = r"D:\supply-chain-analytics\DataCoSupplyChainDataset.csv"
    BATCH_INSERT_SIZE: int = 5000
    
    PROPHET_FORECAST_DAYS: int = 14
    PROPHET_CHANGEPOINT_PRIOR: float = 0.05
    PROPHET_SEASONALITY_PRIOR: float = 10.0
    
    RF_N_ESTIMATORS: int = 200
    RF_MAX_DEPTH: int = 20
    RF_MIN_SAMPLES_SPLIT: int = 5
    
    KMEANS_N_CLUSTERS: int = 5
    KMEANS_MAX_ITER: int = 300
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
        
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

# ============================================================================
# DATABASE SETUP
# ============================================================================

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# ============================================================================
# DATABASE MODELS
# ============================================================================

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, nullable=False, index=True)
    order_date = Column(DateTime(timezone=True), nullable=False, index=True)
    
    customer_id = Column(Integer, nullable=False, index=True)
    customer_country = Column(String(100))
    customer_city = Column(String(100))
    customer_state = Column(String(100))
    customer_segment = Column(String(50))
    
    product_id = Column(Integer, nullable=False, index=True)
    product_name = Column(Text)
    product_category = Column(String(100), index=True)
    product_price = Column(Float)
    
    order_quantity = Column(Integer)
    sales = Column(Float)
    discount = Column(Float)
    profit = Column(Float)
    
    shipping_mode = Column(String(50))
    delivery_status = Column(String(50), index=True)
    days_for_shipping_real = Column(Integer)
    days_for_shipment_scheduled = Column(Integer)
    
    latitude = Column(Float)
    longitude = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=sql_func.now())
    
    __table_args__ = (
        Index('idx_order_date_product', 'order_date', 'product_id'),
        Index('idx_order_date_category', 'order_date', 'product_category'),
    )

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class DashboardMetrics(BaseModel):
    total_orders: int
    total_sales: float
    total_profit: float
    avg_delivery_days: float
    on_time_delivery_rate: float
    total_products: int
    active_customers: int

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

class DataLoader:
    
    @staticmethod
    def load_kaggle_dataset(file_path: str) -> pd.DataFrame:
        """Load DataCo Supply Chain dataset with timezone-aware dates"""
        
        logger.info(f"Attempting to load CSV from: {file_path}")
        
        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at: {file_path}")
        
        # Load CSV
        logger.info("Reading CSV file...")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Parse dates and ADD TIMEZONE
        logger.info("Processing dates with timezone...")
        df['order date (DateOrders)'] = pd.to_datetime(
            df['order date (DateOrders)'], 
            errors='coerce'
        ).dt.tz_localize('UTC')  # Add this line!
        
        # Create clean dataframe
        logger.info("Mapping columns...")
        df_clean = pd.DataFrame({
            'order_id': df['Order Item Id'].astype(int),
            'order_date': df['order date (DateOrders)'],  # Now timezone-aware
            'customer_id': df['Customer Id'].astype(int),
            'customer_country': df['Customer Country'].fillna('Unknown').astype(str),
            'customer_city': df['Customer City'].fillna('Unknown').astype(str),
            'customer_state': df['Customer State'].fillna('Unknown').astype(str),
            'customer_segment': df['Customer Segment'].fillna('Unknown').astype(str),
            'product_id': df['Order Item Cardprod Id'].astype(int),
            'product_name': df['Product Name'].fillna('Unknown').astype(str),
            'product_category': df['Category Name'].fillna('Unknown').astype(str),
            'product_price': pd.to_numeric(df['Order Item Product Price'], errors='coerce').fillna(0),
            'order_quantity': pd.to_numeric(df['Order Item Quantity'], errors='coerce').fillna(1).astype(int),
            'sales': pd.to_numeric(df['Sales'], errors='coerce').fillna(0),
            'discount': pd.to_numeric(df['Order Item Discount'], errors='coerce').fillna(0),
            'profit': pd.to_numeric(df['Benefit per order'], errors='coerce').fillna(0),
            'shipping_mode': df['Shipping Mode'].fillna('Unknown').astype(str),
            'delivery_status': df['Delivery Status'].fillna('Unknown').astype(str),
            'days_for_shipping_real': pd.to_numeric(df['Days for shipping (real)'], errors='coerce').fillna(0).astype(int),
            'days_for_shipment_scheduled': pd.to_numeric(df['Days for shipment (scheduled)'], errors='coerce').fillna(0).astype(int),
            'latitude': pd.to_numeric(df['Latitude'], errors='coerce').fillna(0),
            'longitude': pd.to_numeric(df['Longitude'], errors='coerce').fillna(0)
        })
        
        # Drop rows with invalid dates
        df_clean = df_clean.dropna(subset=['order_date'])
        
        logger.info(f"Cleaned dataset: {len(df_clean)} records")
        logger.info(f"Date range: {df_clean['order_date'].min()} to {df_clean['order_date'].max()}")
        logger.info(f"Unique products: {df_clean['product_id'].nunique()}")
        logger.info(f"Unique customers: {df_clean['customer_id'].nunique()}")
        
        return df_clean
    
    @staticmethod
    async def bulk_insert_orders(df: pd.DataFrame, session: AsyncSession, batch_size: int = 5000):
        """Bulk insert with progress tracking"""
        logger.info(f"Starting bulk insert: {len(df)} records, batch size: {batch_size}")
        
        total_inserted = 0
        total_rows = len(df)
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # Convert batch to list of dicts
            values = batch_df.to_dict('records')
            
            try:
                # Bulk insert
                await session.execute(
                    Order.__table__.insert(),
                    values
                )
                await session.commit()
                
                total_inserted += len(values)
                progress = (total_inserted / total_rows) * 100
                
                logger.info(f"Progress: {total_inserted}/{total_rows} ({progress:.1f}%)")
                
                # Broadcast progress
                await manager.broadcast({
                    'type': 'data_loading_progress',
                    'data': {
                        'total': total_rows,
                        'inserted': total_inserted,
                        'progress': round(progress, 2)
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error inserting batch at index {i}: {str(e)}")
                await session.rollback()
                raise
        
        logger.info(f"Bulk insert completed: {total_inserted} records")
        return total_inserted

# ============================================================================
# MACHINE LEARNING MODELS
# ============================================================================

class DemandForecaster:
    
    def __init__(self):
        self.models: Dict[int, Prophet] = {}
        self.forecast_days = settings.PROPHET_FORECAST_DAYS
        
    def prepare_data(self, df: pd.DataFrame, product_id: int) -> pd.DataFrame:
        product_data = df[df['product_id'] == product_id].copy()
        
        daily_demand = product_data.groupby('order_date').agg({
            'order_quantity': 'sum'
        }).reset_index()
        
        daily_demand.columns = ['ds', 'y']
        daily_demand['ds'] = pd.to_datetime(daily_demand['ds'])
        daily_demand = daily_demand.sort_values('ds')
        
        return daily_demand
    
    def train_model(self, product_id: int, training_data: pd.DataFrame) -> Prophet:
        model = Prophet(
            changepoint_prior_scale=settings.PROPHET_CHANGEPOINT_PRIOR,
            seasonality_prior_scale=settings.PROPHET_SEASONALITY_PRIOR,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95
        )
        
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        model.fit(training_data)
        self.models[product_id] = model
        return model
    
    def forecast_product_demand(self, df: pd.DataFrame, product_id: int) -> pd.DataFrame:
        training_data = self.prepare_data(df, product_id)
        
        if len(training_data) < 14:
            raise ValueError(f"Insufficient data for product {product_id}")
        
        model = self.train_model(product_id, training_data)
        
        future = model.make_future_dataframe(periods=self.forecast_days, freq='D')
        forecast = model.predict(future)
        
        forecast_future = forecast.tail(self.forecast_days)
        
        result = pd.DataFrame({
            'forecast_date': forecast_future['ds'],
            'predicted_demand': forecast_future['yhat'],
            'yhat_lower': forecast_future['yhat_lower'],
            'yhat_upper': forecast_future['yhat_upper'],
            'product_id': product_id
        })
        
        return result
    
    async def forecast_multiple_products(self, df: pd.DataFrame, product_ids: List[int]) -> pd.DataFrame:
        all_forecasts = []
        
        for idx, product_id in enumerate(product_ids):
            try:
                forecast = self.forecast_product_demand(df, product_id)
                all_forecasts.append(forecast)
                
                await manager.broadcast({
                    'type': 'forecast_progress',
                    'data': {
                        'product_id': int(product_id),
                        'completed': idx + 1,
                        'total': len(product_ids)
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error forecasting product {product_id}: {str(e)}")
                continue
        
        if all_forecasts:
            return pd.concat(all_forecasts, ignore_index=True)
        return pd.DataFrame()

class InventoryOptimizer:
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        product_stats = df.groupby('product_id').agg({
            'order_quantity': ['mean', 'std', 'sum', 'count'],
            'sales': ['mean', 'sum'],
            'profit': 'mean',
            'days_for_shipping_real': 'mean',
            'order_date': lambda x: (x.max() - x.min()).days
        }).reset_index()
        
        product_stats.columns = [
            'product_id', 'avg_daily_demand', 'demand_std', 'total_demand', 
            'order_count', 'avg_sales', 'total_sales', 'avg_profit_margin',
            'avg_lead_time', 'days_in_system'
        ]
        
        product_stats['demand_variability'] = (
            product_stats['demand_std'] / product_stats['avg_daily_demand']
        ).fillna(0)
        
        product_stats['daily_demand_rate'] = (
            product_stats['total_demand'] / (product_stats['days_in_system'] + 1)
        ).fillna(0)
        
        product_stats['sales_velocity'] = (
            product_stats['total_sales'] / (product_stats['days_in_system'] + 1)
        ).fillna(0)
        
        return product_stats
    
    def calculate_reorder_metrics(self, stats: pd.DataFrame) -> pd.DataFrame:
        Z_SCORE = 1.65
        
        stats['safety_stock'] = (
            Z_SCORE * stats['demand_std'] * np.sqrt(stats['avg_lead_time'] + 1)
        )
        
        stats['reorder_point'] = (
            stats['avg_daily_demand'] * stats['avg_lead_time'] + stats['safety_stock']
        )
        
        stats['optimal_order_quantity'] = np.sqrt(
            (2 * stats['daily_demand_rate'] * 100) / 
            (0.25 * (stats['avg_sales'] + 1))
        )
        
        stats['stockout_probability'] = np.maximum(
            0, 
            1 - (stats['daily_demand_rate'] / (stats['reorder_point'] + 1))
        )
        
        return stats
    
    def predict_inventory_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = self.calculate_features(df)
        features_df = self.calculate_reorder_metrics(features_df)
        
        result_columns = [
            'product_id', 'reorder_point', 'safety_stock',
            'optimal_order_quantity', 'stockout_probability',
            'avg_daily_demand', 'demand_std', 'avg_lead_time'
        ]
        
        return features_df[result_columns]

class RouteOptimizer:
    
    def __init__(self):
        self.n_clusters = settings.KMEANS_N_CLUSTERS
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_location_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
    def optimize_routes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        location_data = self.prepare_location_data(df)
        
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
            'cluster_center_lon': 'first'
        }).reset_index()
        
        cluster_summary.columns = [
            'cluster_id', 'customer_count', 'total_orders', 'total_sales',
            'avg_delivery_days', 'avg_distance_to_center',
            'cluster_center_lat', 'cluster_center_lon'
        ]
        
        return location_data, cluster_summary

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")
    yield
    logger.info("Shutting down")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'ping':
                await websocket.send_json({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Supply Chain Analytics API",
        "version": "2.0",
        "csv_path": settings.KAGGLE_DATASET_PATH,
        "websocket": "ws://localhost:8000/ws",
        "docs": "http://localhost:8000/docs"
    }

@app.post(f"{settings.API_V1_PREFIX}/data/load-kaggle")
async def load_kaggle_dataset(db: AsyncSession = Depends(get_db)):
    """Load DataCo Supply Chain dataset into database"""
    try:
        loader = DataLoader()
        
        # Load CSV
        df = loader.load_kaggle_dataset(settings.KAGGLE_DATASET_PATH)
        
        # Bulk insert
        total_inserted = await loader.bulk_insert_orders(df, db, settings.BATCH_INSERT_SIZE)
        
        await manager.broadcast({
            'type': 'data_loaded',
            'data': {
                'total_records': total_inserted,
                'status': 'completed'
            },
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'status': 'success',
            'records_loaded': total_inserted,
            'message': f'Successfully loaded {total_inserted} records'
        }
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/dashboard/metrics")
async def get_dashboard_metrics(db: AsyncSession = Depends(get_db)):
    """Get dashboard KPIs"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'order_id': o.order_id,
            'sales': o.sales,
            'profit': o.profit,
            'days_for_shipping_real': o.days_for_shipping_real,
            'days_for_shipment_scheduled': o.days_for_shipment_scheduled,
            'product_id': o.product_id,
            'customer_id': o.customer_id
        } for o in orders])
        
        on_time = len(df[df['days_for_shipping_real'] <= df['days_for_shipment_scheduled']])
        
        metrics = {
            'total_orders': int(len(df)),
            'total_sales': float(df['sales'].sum()),
            'total_profit': float(df['profit'].sum()),
            'avg_delivery_days': float(df['days_for_shipping_real'].mean()),
            'on_time_delivery_rate': float(on_time / len(df) * 100) if len(df) > 0 else 0,
            'total_products': int(df['product_id'].nunique()),
            'active_customers': int(df['customer_id'].nunique())
        }
        
        await manager.broadcast({
            'type': 'metrics_updated',
            'data': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        return metrics
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/forecasts/demand")
async def get_demand_forecasts(
    product_ids: Optional[List[int]] = Query(None),
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Generate Prophet demand forecasts"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        df = pd.DataFrame([{
            'product_id': o.product_id,
            'order_date': o.order_date,
            'order_quantity': o.order_quantity
        } for o in orders])
        
        if product_ids is None:
            top_products = df.groupby('product_id')['order_quantity'].sum().nlargest(limit)
            product_ids = top_products.index.tolist()
        
        forecaster = DemandForecaster()
        forecasts = await forecaster.forecast_multiple_products(df, product_ids[:limit])
        
        return forecasts.to_dict(orient='records')
    
    except Exception as e:
        logger.error(f"Error generating forecasts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/inventory/optimization")
async def get_inventory_optimization(
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Random Forest inventory optimization"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        df = pd.DataFrame([{
            'product_id': o.product_id,
            'order_quantity': o.order_quantity,
            'sales': o.sales,
            'profit': o.profit,
            'days_for_shipping_real': o.days_for_shipping_real,
            'order_date': o.order_date
        } for o in orders])
        
        optimizer = InventoryOptimizer()
        inventory_metrics = optimizer.predict_inventory_metrics(df)
        
        return inventory_metrics.head(limit).to_dict(orient='records')
    
    except Exception as e:
        logger.error(f"Error optimizing inventory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/routes/clusters")
async def get_route_clusters(db: AsyncSession = Depends(get_db)):
    """K-Means route clustering"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        df = pd.DataFrame([{
            'customer_id': o.customer_id,
            'latitude': o.latitude,
            'longitude': o.longitude,
            'customer_city': o.customer_city,
            'customer_state': o.customer_state,
            'order_id': o.order_id,
            'sales': o.sales,
            'days_for_shipping_real': o.days_for_shipping_real
        } for o in orders])
        
        route_optimizer = RouteOptimizer()
        clustered_data, cluster_summary = route_optimizer.optimize_routes(df)
        
        return {
            'clusters': cluster_summary.to_dict(orient='records'),
            'customer_assignments': clustered_data.head(100).to_dict(orient='records')
        }
    
    except Exception as e:
        logger.error(f"Error clustering routes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/analytics/time-series")
async def get_time_series_sales(
    category: Optional[str] = None,
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """Get time-series sales data"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = select(Order).where(Order.order_date >= cutoff_date)
        
        if category:
            query = query.where(Order.product_category == category)
        
        result = await db.execute(query)
        orders = result.scalars().all()
        
        df = pd.DataFrame([{
            'order_date': o.order_date,
            'sales': o.sales
        } for o in orders])
        
        daily_sales = df.groupby('order_date')['sales'].sum().reset_index()
        daily_sales.columns = ['timestamp', 'value']
        
        return daily_sales.to_dict(orient='records')
    
    except Exception as e:
        logger.error(f"Error getting time-series: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/products/top-performing")
async def get_top_products(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """Get top performing products"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        df = pd.DataFrame([{
            'product_id': o.product_id,
            'product_name': o.product_name,
            'product_category': o.product_category,
            'sales': o.sales,
            'profit': o.profit
        } for o in orders])
        
        product_perf = df.groupby(['product_id', 'product_name', 'product_category']).agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        product_perf.columns = [
            'product_id', 'product_name', 'product_category',
            'total_sales', 'total_profit'
        ]
        
        product_perf['profit_margin'] = (
            product_perf['total_profit'] / (product_perf['total_sales'] + 1) * 100
        )
        
        top_products = product_perf.nlargest(limit, 'total_sales')
        
        return top_products.to_dict(orient='records')
    
    except Exception as e:
        logger.error(f"Error getting top products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "csv_exists": os.path.exists(settings.KAGGLE_DATASET_PATH)
    }

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
