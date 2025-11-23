
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
from app.db.session import get_db
from app.services.data_loader import DataLoader
from app.services.websocket import manager
from app.core.config import settings
from app.models.order import Order
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/load-kaggle")
async def load_kaggle_dataset(db: AsyncSession = Depends(get_db)):
    """Load DataCo Supply Chain dataset into database"""
    try:
        loader = DataLoader()
        
        # Load CSV
        df = await loader.load_kaggle_dataset(settings.KAGGLE_DATASET_PATH)
        
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

@router.get("/products")
async def get_products(
    limit: int = 100,
    search: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Get unique products"""
    try:
        query = select(
            Order.product_id, 
            Order.product_name, 
            Order.product_category,
            func.count(Order.order_id).label('total_orders')
        ).group_by(
            Order.product_id, 
            Order.product_name, 
            Order.product_category
        ).order_by(func.count(Order.order_id).desc())
        
        if search:
            query = query.where(Order.product_name.ilike(f"%{search}%"))
            
        query = query.limit(limit)
        
        result = await db.execute(query)
        products = result.all()
        
        return {
            "products": [
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category_name": p.product_category,
                    "total_orders": p.total_orders
                } for p in products
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
