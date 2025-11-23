
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd
import logging

from app.db.session import get_db
from app.models.order import Order
from app.services.anomaly_detection import AnomalyDetector

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/detect")
async def detect_anomalies(
    product_id: int,
    days: int = 90,
    db: AsyncSession = Depends(get_db)
):
    """Detect anomalies for a specific product"""
    try:
        result = await db.execute(select(Order).where(Order.product_id == product_id))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="Product not found or insufficient data")
            
        df = pd.DataFrame([{
            'order_date': o.order_date,
            'order_quantity': o.order_quantity,
            'sales': o.sales,
            'profit': o.profit
        } for o in orders])
        
        detector = AnomalyDetector()
        anomalies = await detector.detect_anomalies(df, days)
        
        return anomalies
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
