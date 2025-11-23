
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd
import logging

from app.db.session import get_db
from app.models.order import Order
from app.services.route_optimization import RouteOptimizer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/clusters")
async def get_route_clusters(
    date: str = None,
    clusters: int = 5,
    db: AsyncSession = Depends(get_db)
):
    """K-Means route clustering"""
    try:
        # In a real app, we would filter by date here
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
        # Service now returns the formatted dict directly
        result_data = await route_optimizer.optimize_routes(df)
        
        return result_data
    
    except Exception as e:
        logger.error(f"Error clustering routes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error clustering routes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
