
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import pandas as pd
import logging

from app.db.session import get_db
from app.models.order import Order
from app.services.demand_forecasting import DemandForecaster

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/demand")
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

from pydantic import BaseModel

class ForecastRequest(BaseModel):
    product_id: int
    days: int = 30
    show_ci: bool = True

@router.post("/generate")
async def generate_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate detailed forecast for a single product"""
    try:
        result = await db.execute(select(Order).where(Order.product_id == request.product_id))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="Product not found or insufficient data")
            
        df = pd.DataFrame([{
            'product_id': o.product_id,
            'order_date': o.order_date,
            'order_quantity': o.order_quantity
        } for o in orders])
        
        forecaster = DemandForecaster()
        forecast_result = await forecaster.forecast_product_demand(df, request.product_id, request.days)
        
        return forecast_result
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
