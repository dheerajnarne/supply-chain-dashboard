"""
Supply Chain Predictive Analytics Platform
Production-ready FastAPI application with real-time forecasting, ML models, and WebSocket streaming
Version: 2.0.0 | Industry-Standard Architecture
"""

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy import PrimaryKeyConstraint, ForeignKeyConstraint, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import OperationalError
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
from contextlib import asynccontextmanager

warnings.filterwarnings('ignore')

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE

# ========== LOGGING CONFIGURATION ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('supply_chain_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
DATABASE_URL = "postgresql://postgres:%40Dheeraj123@localhost:5432/supply_chain_db"
CSV_FILE_PATH = "DataCoSupplyChainDataset.csv"
PLOT_DIR = "forecast_plots"
MODEL_DIR = "trained_models"
CACHE_DIR = "cache"

for directory in [PLOT_DIR, MODEL_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# API Configuration
API_VERSION = "2.0.0"
API_TITLE = "Supply Chain Predictive Analytics Platform"
MAX_FORECAST_DAYS = 90
MIN_FORECAST_DAYS = 7

# ========== DATABASE SCHEMA ==========
Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'
    product_card_id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String)
    category_name = Column(String)
    department_name = Column(String)

class Customer(Base):
    __tablename__ = 'customers'
    customer_id = Column(Integer, primary_key=True, index=True)
    customer_fname = Column(String)
    customer_lname = Column(String)
    customer_city = Column(String)
    customer_state = Column(String)
    customer_zipcode = Column(Integer)
    customer_segment = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)

class Order(Base):
    __tablename__ = 'orders'
    order_id = Column(Integer, index=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id'))
    order_date = Column(DateTime(timezone=True), index=True)
    order_status = Column(String)
    order_city = Column(String)
    order_state = Column(String)
    shipping_mode = Column(String)
    days_for_shipping_real = Column(Integer)
    days_for_shipment_scheduled = Column(Integer)
    __table_args__ = (PrimaryKeyConstraint('order_id', 'order_date'),)

class OrderItem(Base):
    __tablename__ = 'order_items'
    order_item_id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, index=True)
    order_date = Column(DateTime(timezone=True))
    product_card_id = Column(Integer, ForeignKey('products.product_card_id'), index=True)
    order_item_quantity = Column(Integer)
    order_item_product_price = Column(Float)
    order_item_discount = Column(Float)
    sales = Column(Float)
    order_profit_per_order = Column(Float)
    __table_args__ = (ForeignKeyConstraint(['order_id', 'order_date'], ['orders.order_id', 'orders.order_date']),)

# ========== PYDANTIC MODELS (ENHANCED) ==========
class ForecastRequest(BaseModel):
    product_id: int = Field(..., gt=0, description="Product ID to forecast")
    forecast_days: int = Field(14, ge=MIN_FORECAST_DAYS, le=MAX_FORECAST_DAYS)
    include_confidence_intervals: bool = Field(True, description="Include prediction intervals")
    
    @validator('forecast_days')
    def validate_horizon(cls, v):
        valid = [7, 14, 21, 28, 30, 60, 90]
        if v not in valid:
            return min(valid, key=lambda x: abs(x - v))
        return v

class ForecastResponse(BaseModel):
    product_id: int
    product_name: str
    forecast_days: int
    forecast_start_date: str
    forecast_end_date: str
    total_demand: float
    avg_daily_demand: float
    validation_metrics: Dict[str, float]
    daily_forecast: List[Dict[str, Any]]
    eda_insights: Dict[str, Any]
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    plot_url: str
    generated_at: str

class InventoryResponse(BaseModel):
    product_id: int
    product_name: str
    reorder_point: float
    safety_stock: float
    optimal_order_quantity: float
    current_stock_days: int
    stockout_risk: str
    model_confidence: float
    recommendations: List[str]

class AnomalyResponse(BaseModel):
    product_id: int
    analysis_period_days: int
    anomalies_detected: int
    anomaly_rate: float
    severity_distribution: Dict[str, int]
    top_anomalies: List[Dict[str, Any]]
    recommendations: List[str]

class RouteResponse(BaseModel):
    date: str
    total_orders: int
    num_routes: int
    routes: List[Dict[str, Any]]
    optimization_metrics: Dict[str, float]
    estimated_savings: float

class ProductInfo(BaseModel):
    product_card_id: int
    product_name: str
    category_name: str
    department_name: str
    total_orders: int
    date_range: Dict[str, str]
    avg_daily_demand: float
    demand_volatility: float

class DashboardMetrics(BaseModel):
    total_products: int
    total_orders: int
    total_revenue: float
    avg_forecast_accuracy: float
    products_at_risk: int
    active_routes: int
    last_updated: str

# ========== WEBSOCKET CONNECTION MANAGER ==========
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = defaultdict(list)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id].append(websocket)
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections[client_id])}")
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        if websocket in self.active_connections[client_id]:
            self.active_connections[client_id].remove(websocket)
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def broadcast(self, message: dict, client_id: str):
        for connection in self.active_connections[client_id]:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ========== ML MODELS ==========
class InventoryOptimizer:
    def __init__(self):
        self.reorder_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.safety_stock_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.is_trained = False
    
    def prepare_features(self, df):
        df = df.copy()
        df['avg_demand_7d'] = df['y'].rolling(window=7, min_periods=1).mean()
        df['std_demand_7d'] = df['y'].rolling(window=7, min_periods=1).std()
        df['avg_demand_30d'] = df['y'].rolling(window=30, min_periods=1).mean()
        df['demand_volatility'] = df['std_demand_7d'] / (df['avg_demand_7d'] + 1e-6)
        df['trend'] = df['y'].pct_change()
        df['day_of_week'] = pd.to_datetime(df['ds']).dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        feature_cols = ['avg_demand_7d', 'std_demand_7d', 'avg_demand_30d', 
                       'demand_volatility', 'trend', 'day_of_week', 'is_weekend']
        return df[feature_cols].fillna(0)
    
    def train(self, historical_data):
        if len(historical_data) < 30:
            raise ValueError("Need at least 30 days")
        
        X = self.prepare_features(historical_data)
        y_reorder = historical_data['y'].rolling(14, min_periods=1).mean() + \
                   1.65 * historical_data['y'].rolling(14, min_periods=1).std()
        y_safety = 1.96 * historical_data['y'].rolling(14, min_periods=1).std()
        
        valid_mask = ~(y_reorder.isna() | y_safety.isna())
        self.reorder_model.fit(X[valid_mask], y_reorder[valid_mask].fillna(0))
        self.safety_stock_model.fit(X[valid_mask], y_safety[valid_mask].fillna(0))
        self.is_trained = True
    
    def predict(self, recent_data):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X = self.prepare_features(recent_data).tail(1)
        reorder = max(0, float(self.reorder_model.predict(X)[0]))
        safety = max(0, float(self.safety_stock_model.predict(X)[0]))
        eoq = float(np.sqrt(2 * reorder * 100 / 5))  # Economic Order Quantity
        
        return {
            'reorder_point': reorder,
            'safety_stock': safety,
            'optimal_order_quantity': eoq,
            'confidence': 0.85
        }

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    
    def detect(self, df):
        df = df.copy()
        df['day_of_week'] = pd.to_datetime(df['ds']).dt.dayofweek
        df['rolling_mean'] = df['y'].rolling(7, min_periods=1).mean()
        df['rolling_std'] = df['y'].rolling(7, min_periods=1).std()
        df['deviation'] = (df['y'] - df['rolling_mean']) / (df['rolling_std'] + 1e-6)
        
        features = df[['y', 'day_of_week', 'deviation']].fillna(0).values
        df['is_anomaly'] = self.model.fit_predict(features) == -1
        df['severity'] = -self.model.score_samples(features)
        
        anomalies = df[df['is_anomaly']].copy()
        anomalies['severity_level'] = pd.cut(anomalies['severity'], 
                                             bins=[0, 0.5, 0.7, 1.0],
                                             labels=['Low', 'Medium', 'High'])
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df) * 100,
            'severity_counts': anomalies['severity_level'].value_counts().to_dict(),
            'top_anomalies': anomalies.nlargest(5, 'severity')[
                ['ds', 'y', 'rolling_mean', 'severity_level']
            ].to_dict('records')
        }

class RouteOptimizer:
    def __init__(self, n_routes=5):
        self.kmeans = KMeans(n_clusters=n_routes, random_state=42, n_init=10)
    
    def optimize(self, orders_df):
        coords = orders_df[['latitude', 'longitude']].values
        valid = ~np.isnan(coords).any(axis=1)
        
        orders_clean = orders_df[valid].copy()
        orders_clean['route'] = self.kmeans.fit_predict(coords[valid])
        
        routes = []
        for i in range(self.kmeans.n_clusters):
            cluster = orders_clean[orders_clean['route'] == i]
            routes.append({
                'route_id': i,
                'num_orders': len(cluster),
                'total_demand': float(cluster['order_item_quantity'].sum()),
                'cities': cluster['order_city'].unique().tolist()[:5]
            })
        
        return {
            'routes': routes,
            'balance_score': 1 - np.std([r['total_demand'] for r in routes]) / 
                           (np.mean([r['total_demand'] for r in routes]) + 1)
        }

# ========== HELPER FUNCTIONS ==========
def get_db():
    """Database dependency"""
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def setup_database(engine):
    """Create tables and hypertable"""
    logger.info("Setting up database...")
    try:
        Base.metadata.create_all(engine)
        logger.info("[OK] Tables created")  # Changed
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False
    
    with engine.connect() as conn:
        try:
            conn.execute(text("SELECT create_hypertable('orders', 'order_date');"))
            conn.commit()
            logger.info("[OK] Hypertable created")  # Changed
        except Exception as e:
            if "already a hypertable" in str(e).lower():
                logger.info("[OK] Hypertable exists")  # Changed
                conn.rollback()
            else:
                logger.error(f"Hypertable error: {e}")
                conn.rollback()
    return True

def get_product_demand(engine, product_id: int):
    """Fetch demand history"""
    query = text("""
        SELECT DATE_TRUNC('day', order_date) AS ds, SUM(order_item_quantity) AS y
        FROM order_items WHERE product_card_id = :product_id
        GROUP BY ds ORDER BY ds;
    """)
    df = pd.read_sql(query, engine, params={'product_id': product_id})
    df['ds'] = pd.to_datetime(df['ds'])
    return df.sort_values('ds').reset_index(drop=True)

def get_product_name(engine, product_id: int) -> str:
    """Get product name"""
    query = text("SELECT product_name FROM products WHERE product_card_id = :id")
    with engine.connect() as conn:
        result = conn.execute(query, {"id": product_id}).fetchone()
        return result[0] if result else f"Product {product_id}"

# ========== FORECASTING (ENHANCED) ==========
def generate_forecast_enhanced(engine, product_id: int, forecast_days: int, include_ci: bool = True):
    """Enhanced forecasting with confidence intervals"""
    
    SEQ_LEN, PRED_LEN, TRAIN_RATIO = 56, 14, 0.7
    
    df = get_product_demand(engine, product_id)
    if len(df) < SEQ_LEN + PRED_LEN:
        raise ValueError(f"Need {SEQ_LEN + PRED_LEN} days, got {len(df)}")
    
    # Fill missing dates
    date_range = pd.date_range(df['ds'].min(), df['ds'].max(), freq='D')
    full_df = pd.DataFrame({'ds': date_range})
    df = full_df.merge(df, on='ds', how='left').fillna(0)
    df['unique_id'] = f'product_{product_id}'
    df = df[['unique_id', 'ds', 'y']]
    
    # Train/test split
    train_size = int(len(df) * TRAIN_RATIO)
    train_df, test_df = df[:train_size].copy(), df[train_size:].copy()
    
    # Validation
    validation_model = NHITS(
        h=PRED_LEN, input_size=SEQ_LEN, loss=MAE(), max_steps=200,
        val_check_steps=50, early_stop_patience_steps=3, learning_rate=1e-3,
        stack_types=['identity'] * 3, n_blocks=[1] * 3,
        mlp_units=[[128, 128]] * 3, n_pool_kernel_size=[2, 2, 1],
        n_freq_downsample=[4, 2, 1], pooling_mode='MaxPool1d',
        interpolation_mode='linear', batch_size=32, random_seed=42, scaler_type='robust'
    )
    
    nf_val = NeuralForecast(models=[validation_model], freq='D')
    nf_val.fit(df=train_df, val_size=56)
    
    # Calculate metrics
    preds, acts = [], []
    for i in range((len(test_df) - PRED_LEN) // PRED_LEN + 1):
        start = i * PRED_LEN
        if start + PRED_LEN > len(test_df):
            break
        input_df = df[:train_size + start].copy()
        if len(input_df) >= SEQ_LEN:
            try:
                fore = nf_val.predict(df=input_df)
                preds.extend(fore['NHITS'].values[:PRED_LEN])
                acts.extend(test_df.iloc[start:start+PRED_LEN]['y'].values)
            except:
                continue
    
    preds, acts = np.array(preds), np.array(acts)
    mae = mean_absolute_error(acts, preds)
    rmse = np.sqrt(mean_squared_error(acts, preds))
    mape = np.mean(np.abs((acts[acts!=0] - preds[acts!=0]) / acts[acts!=0])) * 100
    
    # Future forecast
    if forecast_days != PRED_LEN:
        future_model = NHITS(
            h=forecast_days, input_size=SEQ_LEN, loss=MAE(), max_steps=200,
            val_check_steps=50, early_stop_patience_steps=3, learning_rate=1e-3,
            stack_types=['identity'] * 3, n_blocks=[1] * 3,
            mlp_units=[[128, 128]] * 3, n_pool_kernel_size=[2, 2, 1],
            n_freq_downsample=[4, 2, 1], pooling_mode='MaxPool1d',
            interpolation_mode='linear', batch_size=32, random_seed=42, scaler_type='robust'
        )
        nf_future = NeuralForecast(models=[future_model], freq='D')
        nf_future.fit(df=df, val_size=56)
    else:
        nf_future = nf_val
    
    forecast = nf_future.predict(df=df)
    predictions = forecast['NHITS'].values
    
    last_date = df['ds'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    
    # Confidence intervals
    ci_lower, ci_upper = None, None
    if include_ci:
        std_error = rmse
        ci_lower = predictions - 1.96 * std_error
        ci_upper = predictions + 1.96 * std_error
    
    # EDA
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'demand': predictions,
        'day_of_week': future_dates.day_name(),
        'is_weekend': future_dates.dayofweek >= 5
    })
    
    slope, _, _, _, _ = stats.linregress(np.arange(len(predictions)), predictions)
    
    # Generate plot
    plot_path = generate_enhanced_plot(df, future_dates, predictions, ci_lower, ci_upper, 
                                       product_id, {'mae': mae, 'rmse': rmse, 'mape': mape})
    
    return {
        "product_id": product_id,
        "product_name": get_product_name(engine, product_id),
        "forecast_days": forecast_days,
        "forecast_start_date": future_dates[0].strftime('%Y-%m-%d'),
        "forecast_end_date": future_dates[-1].strftime('%Y-%m-%d'),
        "total_demand": float(predictions.sum()),
        "avg_daily_demand": float(predictions.mean()),
        "validation_metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "accuracy": float(100 - mape)
        },
        "daily_forecast": [
            {
                "date": d.strftime('%Y-%m-%d'),
                "day": d.strftime('%A'),
                "demand": float(p),
                "lower_bound": float(ci_lower[i]) if ci_lower is not None else None,
                "upper_bound": float(ci_upper[i]) if ci_upper is not None else None
            }
            for i, (d, p) in enumerate(zip(future_dates, predictions))
        ],
        "eda_insights": {
            "mean": float(forecast_df['demand'].mean()),
            "trend_direction": "increasing" if slope > 0 else "decreasing",
            "weekend_avg": float(forecast_df[forecast_df['is_weekend']]['demand'].mean()),
            "weekday_avg": float(forecast_df[~forecast_df['is_weekend']]['demand'].mean())
        },
        "confidence_intervals": {
            "lower": ci_lower.tolist() if ci_lower is not None else None,
            "upper": ci_upper.tolist() if ci_upper is not None else None
        } if include_ci else None,
        "plot_url": f"/plot/{os.path.basename(plot_path)}",
        "generated_at": datetime.now().isoformat()
    }

def generate_enhanced_plot(df, future_dates, predictions, ci_lower, ci_upper, product_id, metrics):
    """Enhanced plot with confidence intervals"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Forecast with CI
    ax = axes[0, 0]
    hist_days = min(90, len(df))
    ax.plot(df['ds'][-hist_days:], df['y'][-hist_days:], 
            label='Historical', marker='o', alpha=0.7, linewidth=2, markersize=3, color='#2E86AB')
    ax.plot(future_dates, predictions, label='Forecast', 
            marker='s', alpha=0.9, linewidth=2.5, markersize=4, color='#F18F01', linestyle='--')
    
    if ci_lower is not None:
        ax.fill_between(future_dates, ci_lower, ci_upper, alpha=0.2, color='#F18F01', label='95% CI')
    
    ax.axvline(df['ds'].max(), color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_title(f"Demand Forecast - Product {product_id}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Demand", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Distribution
    ax = axes[0, 1]
    ax.hist(predictions, bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
    ax.axvline(predictions.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {predictions.mean():.1f}')
    ax.set_title("Forecast Distribution", fontsize=13, fontweight='bold')
    ax.set_xlabel("Demand", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Daily breakdown
    ax = axes[1, 0]
    colors = ['#e74c3c' if future_dates[i].dayofweek >= 5 else '#3498db' 
              for i in range(len(predictions))]
    ax.bar(range(len(predictions)), predictions, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title("Daily Forecast Breakdown", fontsize=13, fontweight='bold')
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Demand", fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Metrics
    ax = axes[1, 1]
    ax.axis('off')
    metrics_text = f"""
    VALIDATION METRICS
    ═══════════════════
    MAE:        {metrics['mae']:.2f}
    RMSE:       {metrics['rmse']:.2f}
    MAPE:       {metrics['mape']:.2f}%
    Accuracy:   {100 - metrics['mape']:.1f}%
    
    FORECAST SUMMARY
    ═══════════════════
    Total:      {predictions.sum():.0f} units
    Avg/Day:    {predictions.mean():.1f} units
    Min:        {predictions.min():.1f} units
    Max:        {predictions.max():.1f} units
    """
    ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.8))
    
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f'forecast_{product_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path

# ========== FASTAPI APP WITH LIFESPAN ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Supply Chain Analytics API...")
    engine = create_engine(DATABASE_URL)
    setup_database(engine)
    logger.info("[OK] API Ready")  # Changed
    yield
    # Shutdown
    logger.info("Shutting down API...")

app = FastAPI(
    title=API_TITLE,
    description="Production-grade supply chain analytics with ML forecasting, real-time WebSocket updates, and advanced analytics",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database engine
engine = create_engine(DATABASE_URL)

# ========== WEBSOCKET ENDPOINTS ==========
@app.websocket("/ws/forecast/{product_id}")
async def websocket_forecast(websocket: WebSocket, product_id: int):
    """Real-time forecast streaming"""
    await manager.connect(websocket, f"forecast_{product_id}")
    try:
        while True:
            # Generate fresh forecast
            try:
                forecast = generate_forecast_enhanced(engine, product_id, 14, include_ci=False)
                await manager.send_personal_message({
                    "type": "forecast_update",
                    "data": forecast,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
            except Exception as e:
                await manager.send_personal_message({
                    "type": "error",
                    "message": str(e)
                }, websocket)
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, f"forecast_{product_id}")

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """Real-time dashboard metrics"""
    await manager.connect(websocket, "dashboard")
    try:
        while True:
            query = text("""
                SELECT 
                    COUNT(DISTINCT p.product_card_id) as total_products,
                    COUNT(DISTINCT o.order_id) as total_orders,
                    SUM(oi.sales) as total_revenue
                FROM products p
                LEFT JOIN order_items oi ON p.product_card_id = oi.product_card_id
                LEFT JOIN orders o ON oi.order_id = o.order_id
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query).fetchone()
            
            await manager.send_personal_message({
                "type": "dashboard_update",
                "metrics": {
                    "total_products": result[0],
                    "total_orders": result[1],
                    "total_revenue": float(result[2] or 0)
                },
                "timestamp": datetime.now().isoformat()
            }, websocket)
            
            await asyncio.sleep(10)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, "dashboard")

# ========== REST API ENDPOINTS (ENHANCED) ==========
@app.get("/", tags=["Root"])
async def root():
    """API information"""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "operational",
        "features": [
            "N-HiTS Demand Forecasting with 95% CI",
            "Random Forest Inventory Optimization",
            "K-Means Route Clustering",
            "Isolation Forest Anomaly Detection",
            "Real-time WebSocket Streaming",
            "Industry-Standard REST API"
        ],
        "endpoints": {
            "forecast": "/forecast (POST)",
            "inventory": "/inventory/optimize/{product_id}",
            "routes": "/routes/optimize?date=YYYY-MM-DD",
            "anomalies": "/anomalies/detect/{product_id}",
            "products": "/products",
            "dashboard": "/dashboard/metrics",
            "websocket_forecast": "/ws/forecast/{product_id}",
            "websocket_dashboard": "/ws/dashboard"
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
    }

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def create_forecast(request: ForecastRequest, background_tasks: BackgroundTasks):
    """
    Generate ML-powered demand forecast with confidence intervals
    
    **Features:**
    - N-HiTS neural network forecasting
    - 70/30 train-test validation
    - 95% confidence intervals
    - Comprehensive EDA insights
    """
    try:
        logger.info(f"Forecast requested: Product {request.product_id}, {request.forecast_days} days")
        result = generate_forecast_enhanced(
            engine, 
            request.product_id, 
            request.forecast_days, 
            request.include_confidence_intervals
        )
        logger.info(f"Forecast complete: MAPE={result['validation_metrics']['mape']:.2f}%")
        return result
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inventory/optimize/{product_id}", response_model=InventoryResponse, tags=["Inventory"])
async def optimize_inventory(product_id: int):
    """
    Optimize inventory levels using Random Forest
    
    **Calculates:**
    - Reorder point (95% service level)
    - Safety stock
    - Economic Order Quantity (EOQ)
    """
    try:
        df = get_product_demand(engine, product_id)
        if len(df) < 30:
            raise HTTPException(400, "Need 30+ days of data")
        
        optimizer = InventoryOptimizer()
        optimizer.train(df)
        result = optimizer.predict(df)
        
        # Risk assessment
        recent_avg = df.tail(7)['y'].mean()
        risk = "Low" if recent_avg < result['reorder_point'] * 0.5 else \
               "Medium" if recent_avg < result['reorder_point'] * 0.8 else "High"
        
        recommendations = []
        if risk == "High":
            recommendations.append(f"⚠️ Order {result['optimal_order_quantity']:.0f} units immediately")
        recommendations.append(f"Maintain safety stock of {result['safety_stock']:.0f} units")
        
        return {
            "product_id": product_id,
            "product_name": get_product_name(engine, product_id),
            "reorder_point": result['reorder_point'],
            "safety_stock": result['safety_stock'],
            "optimal_order_quantity": result['optimal_order_quantity'],
            "current_stock_days": int(result['reorder_point'] / (recent_avg + 1)),
            "stockout_risk": risk,
            "model_confidence": result['confidence'],
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/anomalies/detect/{product_id}", response_model=AnomalyResponse, tags=["Anomaly Detection"])
async def detect_anomalies(product_id: int, days: int = Query(90, ge=14, le=365)):
    """
    Detect demand anomalies using Isolation Forest
    
    **Identifies:**
    - Unusual demand spikes/drops
    - Severity levels (Low/Medium/High)
    - Pattern deviations
    """
    try:
        df = get_product_demand(engine, product_id).tail(days)
        if len(df) < 14:
            raise HTTPException(400, "Need 14+ days")
        
        detector = AnomalyDetector()
        result = detector.detect(df)
        
        recommendations = []
        if result['anomaly_rate'] > 15:
            recommendations.append("⚠️ High anomaly rate detected - review demand drivers")
        if len([a for a in result['top_anomalies'] if a.get('severity_level') == 'High']) > 0:
            recommendations.append("🔴 Critical anomalies found - investigate supply disruptions")
        
        return {
            "product_id": product_id,
            "analysis_period_days": days,
            "anomalies_detected": result['total_anomalies'],
            "anomaly_rate": result['anomaly_rate'],
            "severity_distribution": result['severity_counts'],
            "top_anomalies": [
                {
                    "date": str(a['ds'].date()) if isinstance(a['ds'], pd.Timestamp) else a['ds'],
                    "demand": float(a['y']),
                    "expected": float(a['rolling_mean']),
                    "severity": a['severity_level']
                }
                for a in result['top_anomalies']
            ],
            "recommendations": recommendations if recommendations else ["✓ No major anomalies detected"]
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/routes/optimize", response_model=RouteResponse, tags=["Route Optimization"])
async def optimize_routes(date: str = Query(..., description="YYYY-MM-DD"), n_routes: int = Query(5, ge=2, le=20)):
    """
    Optimize delivery routes using K-Means clustering
    """
    try:
        query = text("""
            SELECT o.order_id, o.order_city, o.order_state,
                   c.latitude, c.longitude, oi.order_item_quantity
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE DATE(o.order_date) = :date
        """)
        
        df = pd.read_sql(query, engine, params={'date': date})
        if len(df) < n_routes:
            raise HTTPException(400, f"Need {n_routes}+ orders")
        
        optimizer = RouteOptimizer(n_routes)
        result = optimizer.optimize(df)
        
        savings = result['balance_score'] * 20  # Estimated % savings
        
        return {
            "date": date,
            "total_orders": len(df),
            "num_routes": n_routes,
            "routes": result['routes'],
            "optimization_metrics": {
                "balance_score": result['balance_score'],
                "avg_orders_per_route": len(df) / n_routes
            },
            "estimated_savings": savings
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/products", tags=["Products"])
async def list_products(limit: int = Query(50, ge=1, le=500)):
    """List all products with analytics"""
    query = text("""
        SELECT p.product_card_id, p.product_name, p.category_name, p.department_name,
               COUNT(DISTINCT oi.order_id) as total_orders,
               MIN(oi.order_date) as first_order, MAX(oi.order_date) as last_order,
               AVG(oi.order_item_quantity) as avg_demand,
               STDDEV(oi.order_item_quantity) as demand_std
        FROM products p
        JOIN order_items oi ON p.product_card_id = oi.product_card_id
        GROUP BY p.product_card_id, p.product_name, p.category_name, p.department_name
        HAVING COUNT(DISTINCT oi.order_id) > 10
        ORDER BY total_orders DESC
        LIMIT :limit
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"limit": limit})
        products = [
            {
                "product_card_id": row[0],
                "product_name": row[1],
                "category_name": row[2],
                "department_name": row[3],
                "total_orders": row[4],
                "date_range": {
                    "start": row[5].strftime('%Y-%m-%d'),
                    "end": row[6].strftime('%Y-%m-%d')
                },
                "avg_daily_demand": float(row[7] or 0),
                "demand_volatility": float(row[8] / (row[7] + 1) if row[7] else 0)
            }
            for row in result
        ]
    
    return {"products": products, "count": len(products)}

@app.get("/dashboard/metrics", response_model=DashboardMetrics, tags=["Dashboard"])
async def get_dashboard_metrics():
    """Get real-time dashboard KPIs"""
    query = text("""
        SELECT 
            COUNT(DISTINCT p.product_card_id) as products,
            COUNT(DISTINCT o.order_id) as orders,
            COALESCE(SUM(oi.sales), 0) as revenue
        FROM products p
        LEFT JOIN order_items oi ON p.product_card_id = oi.product_card_id
        LEFT JOIN orders o ON oi.order_id = o.order_id
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
    
    return {
        "total_products": result[0],
        "total_orders": result[1],
        "total_revenue": float(result[2]),
        "avg_forecast_accuracy": 75.5,  # Placeholder - calculate from actual forecasts
        "products_at_risk": 12,  # Placeholder
        "active_routes": 45,  # Placeholder
        "last_updated": datetime.now().isoformat()
    }

@app.get("/plot/{filename}", tags=["Visualization"])
async def get_plot(filename: str):
    """Retrieve forecast plot"""
    path = os.path.join(PLOT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Plot not found")
    return FileResponse(path)

@app.get("/health", tags=["System"])
async def health_check():
    """System health check"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "version": API_VERSION,
        "database": db_status,
        "timestamp": datetime.now().isoformat(),
        "components": {
            "forecasting": "N-HiTS",
            "inventory": "Random Forest",
            "routing": "K-Means",
            "anomaly": "Isolation Forest"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="info")
