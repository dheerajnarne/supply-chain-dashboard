
import pandas as pd
import os
import logging
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.order import Order
from app.services.websocket import manager
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

class DataLoader:
    
    @staticmethod
    def _load_csv_sync(file_path: str) -> pd.DataFrame:
        """Synchronous CSV loading logic"""
        logger.info(f"Attempting to load CSV from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at: {file_path}")
        
        logger.info("Reading CSV file...")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        df.columns = df.columns.str.strip()
        
        logger.info("Processing dates with timezone...")
        df['order date (DateOrders)'] = pd.to_datetime(
            df['order date (DateOrders)'], 
            errors='coerce'
        ).dt.tz_localize('UTC')
        
        logger.info("Mapping columns...")
        df_clean = pd.DataFrame({
            'order_id': df['Order Item Id'].astype(int),
            'order_date': df['order date (DateOrders)'],
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
        
        df_clean = df_clean.dropna(subset=['order_date'])
        
        logger.info(f"Cleaned dataset: {len(df_clean)} records")
        return df_clean

    @staticmethod
    async def load_kaggle_dataset(file_path: str) -> pd.DataFrame:
        """Load DataCo Supply Chain dataset with timezone-aware dates (Non-blocking)"""
        return await run_in_threadpool(DataLoader._load_csv_sync, file_path)
    
    @staticmethod
    async def bulk_insert_orders(df: pd.DataFrame, session: AsyncSession, batch_size: int = 5000):
        """Bulk insert with progress tracking"""
        logger.info(f"Starting bulk insert: {len(df)} records, batch size: {batch_size}")
        
        total_inserted = 0
        total_rows = len(df)
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # Convert batch to list of dicts (CPU bound, but usually fast enough for small batches)
            # If this becomes a bottleneck, we can move it to threadpool too
            values = await run_in_threadpool(lambda: batch_df.to_dict('records'))
            
            try:
                await session.execute(
                    Order.__table__.insert(),
                    values
                )
                await session.commit()
                
                total_inserted += len(values)
                progress = (total_inserted / total_rows) * 100
                
                logger.info(f"Progress: {total_inserted}/{total_rows} ({progress:.1f}%)")
                
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
