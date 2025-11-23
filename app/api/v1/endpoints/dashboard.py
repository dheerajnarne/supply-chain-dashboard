
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import pandas as pd
import logging

from app.db.session import get_db
from app.models.order import Order
from app.services.websocket import manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/metrics")
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
        
        # Calculate additional metrics
        on_time = len(df[df['days_for_shipping_real'] <= df['days_for_shipment_scheduled']])
        total_revenue = float(df['sales'].sum())
        total_profit = float(df['profit'].sum())
        
        metrics = {
            'total_orders': int(len(df)),
            'total_sales': total_revenue,
            'total_profit': total_profit,
            'avg_profit': float(df['profit'].mean()),
            'profit_margin': float((total_profit / total_revenue * 100) if total_revenue > 0 else 0),
            'avg_delivery_days': float(df['days_for_shipping_real'].mean()),
            'on_time_delivery_rate': float(on_time / len(df) * 100) if len(df) > 0 else 0,
            'total_products': int(df['product_id'].nunique()),
            'active_customers': int(df['customer_id'].nunique()),
            'avg_order_value': float(df['sales'].mean())
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
