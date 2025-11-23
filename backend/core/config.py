
from pydantic_settings import BaseSettings
from functools import lru_cache

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
    
    # Auth
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
