
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd
import logging

from app.db.session import get_db
from app.models.order import Order
from app.services.inventory_optimization import InventoryOptimizer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/optimization")
async def get_inventory_optimization(
    limit: int = 50,
    product_id: int = None,
    db: AsyncSession = Depends(get_db)
):
    """Random Forest inventory optimization"""
    try:
        query = select(Order)
        if product_id:
            query = query.where(Order.product_id == product_id)
            
        result = await db.execute(query)
        orders = result.scalars().all()
        
        if not orders:
            if product_id:
                raise HTTPException(status_code=404, detail="Product not found or no orders")
            return []
        
        df = pd.DataFrame([{
            'product_id': o.product_id,
            'order_quantity': o.order_quantity,
            'sales': o.sales,
            'profit': o.profit,
            'days_for_shipping_real': o.days_for_shipping_real,
            'order_date': o.order_date
        } for o in orders])
        
        optimizer = InventoryOptimizer()
        inventory_metrics = await optimizer.predict_inventory_metrics(df)
        
        if product_id:
            if inventory_metrics.empty:
                 raise HTTPException(status_code=404, detail="Could not calculate metrics")
            return inventory_metrics.iloc[0].to_dict()
            
        return inventory_metrics.head(limit).to_dict(orient='records')
    
    except Exception as e:
        logger.error(f"Error optimizing inventory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
