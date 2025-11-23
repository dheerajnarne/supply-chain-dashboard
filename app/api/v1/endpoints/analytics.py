from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from typing import Optional

from app.db.session import get_db
from app.models.order import Order

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/sales-over-time")
async def get_sales_over_time(
    granularity: str = Query("monthly", regex="^(daily|weekly|monthly|quarterly)$"),
    db: AsyncSession = Depends(get_db)
):
    """Get sales trends over time"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'order_date': o.order_date,
            'sales': o.sales,
            'profit': o.profit,
            'quantity': o.order_quantity
        } for o in orders])
        
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Resample based on granularity
        freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'quarterly': 'Q'}
        freq = freq_map[granularity]
        
        time_series = df.set_index('order_date').resample(freq).agg({
            'sales': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        return {
            'labels': time_series['order_date'].dt.strftime('%Y-%m-%d').tolist(),
            'sales': time_series['sales'].tolist(),
            'profit': time_series['profit'].tolist(),
            'quantity': time_series['quantity'].tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sales over time: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sales-by-category")
async def get_sales_by_category(db: AsyncSession = Depends(get_db)):
    """Get sales breakdown by product category"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'category': o.product_category,
            'sales': o.sales,
            'profit': o.profit,
            'quantity': o.order_quantity
        } for o in orders])
        
        category_stats = df.groupby('category').agg({
            'sales': ['sum', 'mean', 'count'],
            'profit': ['sum', 'mean'],
            'quantity': 'sum'
        }).reset_index()
        
        category_stats.columns = ['category', 'total_sales', 'avg_sales', 'order_count', 
                                   'total_profit', 'avg_profit', 'total_quantity']
        
        # Calculate profit margin
        category_stats['profit_margin'] = (
            category_stats['total_profit'] / category_stats['total_sales'] * 100
        )
        
        return {
            'categories': category_stats['category'].tolist(),
            'total_sales': category_stats['total_sales'].tolist(),
            'total_profit': category_stats['total_profit'].tolist(),
            'order_count': category_stats['order_count'].tolist(),
            'profit_margin': category_stats['profit_margin'].tolist(),
            'avg_sales': category_stats['avg_sales'].tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sales by category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sales-by-region")
async def get_sales_by_region(db: AsyncSession = Depends(get_db)):
    """Get sales breakdown by region/country"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'country': o.customer_country,
            'state': o.customer_state,
            'city': o.customer_city,
            'sales': o.sales,
            'profit': o.profit
        } for o in orders])
        
        # Country-level analysis
        country_stats = df.groupby('country').agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index().sort_values('sales', ascending=False)
        
        # Top cities analysis
        city_stats = df.groupby(['country', 'city']).agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index().sort_values('sales', ascending=False).head(20)
        
        return {
            'countries': {
                'names': country_stats['country'].tolist(),
                'sales': country_stats['sales'].tolist(),
                'profit': country_stats['profit'].tolist()
            },
            'top_cities': {
                'cities': city_stats['city'].tolist(),
                'countries': city_stats['country'].tolist(),
                'sales': city_stats['sales'].tolist(),
                'profit': city_stats['profit'].tolist()
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sales by region: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customer-segments")
async def get_customer_segments(db: AsyncSession = Depends(get_db)):
    """Get customer segment analysis"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'segment': o.customer_segment,
            'sales': o.sales,
            'profit': o.profit,
            'discount': o.discount
        } for o in orders])
        
        segment_stats = df.groupby('segment').agg({
            'sales': ['sum', 'mean', 'count'],
            'profit': 'sum',
            'discount': 'mean'
        }).reset_index()
        
        segment_stats.columns = ['segment', 'total_sales', 'avg_sales', 'order_count', 
                                  'total_profit', 'avg_discount']
        
        return {
            'segments': segment_stats['segment'].tolist(),
            'total_sales': segment_stats['total_sales'].tolist(),
            'avg_sales': segment_stats['avg_sales'].tolist(),
            'order_count': segment_stats['order_count'].tolist(),
            'total_profit': segment_stats['total_profit'].tolist(),
            'avg_discount': (segment_stats['avg_discount'] * 100).tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer segments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shipping-performance")
async def get_shipping_performance(db: AsyncSession = Depends(get_db)):
    """Get shipping mode performance analysis"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'shipping_mode': o.shipping_mode,
            'days_real': o.days_for_shipping_real,
            'days_scheduled': o.days_for_shipment_scheduled,
            'sales': o.sales
        } for o in orders])
        
        df['on_time'] = df['days_real'] <= df['days_scheduled']
        df['delay_days'] = df['days_real'] - df['days_scheduled']
        
        shipping_stats = df.groupby('shipping_mode').agg({
            'on_time': 'mean',
            'days_real': 'mean',
            'days_scheduled': 'mean',
            'delay_days': 'mean',
            'sales': 'count'
        }).reset_index()
        
        return {
            'shipping_modes': shipping_stats['shipping_mode'].tolist(),
            'on_time_rate': (shipping_stats['on_time'] * 100).tolist(),
            'avg_delivery_days': shipping_stats['days_real'].tolist(),
            'avg_scheduled_days': shipping_stats['days_scheduled'].tolist(),
            'avg_delay': shipping_stats['delay_days'].tolist(),
            'order_count': shipping_stats['sales'].tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting shipping performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/product-performance")
async def get_product_performance(
    top_n: int = Query(20, ge=5, le=100),
    sort_by: str = Query("sales", regex="^(sales|profit|quantity)$"),
    db: AsyncSession = Depends(get_db)
):
    """Get top performing products"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'product_id': o.product_id,
            'product_name': o.product_name,
            'category': o.product_category,
            'sales': o.sales,
            'profit': o.profit,
            'quantity': o.order_quantity,
            'price': o.product_price
        } for o in orders])
        
        product_stats = df.groupby(['product_id', 'product_name', 'category']).agg({
            'sales': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'price': 'mean'
        }).reset_index()
        
        # Calculate metrics
        product_stats['profit_margin'] = (
            product_stats['profit'] / product_stats['sales'] * 100
        )
        product_stats['avg_order_value'] = product_stats['sales'] / product_stats['quantity']
        
        # Sort and get top N
        product_stats = product_stats.sort_values(sort_by, ascending=False).head(top_n)
        
        return {
            'products': product_stats['product_name'].tolist(),
            'categories': product_stats['category'].tolist(),
            'total_sales': product_stats['sales'].tolist(),
            'total_profit': product_stats['profit'].tolist(),
            'total_quantity': product_stats['quantity'].tolist(),
            'profit_margin': product_stats['profit_margin'].tolist(),
            'avg_price': product_stats['price'].tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profit-analysis")
async def get_profit_analysis(db: AsyncSession = Depends(get_db)):
    """Get detailed profit analysis"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'order_date': o.order_date,
            'sales': o.sales,
            'profit': o.profit,
            'discount': o.discount,
            'category': o.product_category
        } for o in orders])
        
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['profit_margin'] = (df['profit'] / df['sales'] * 100)
        
        # Create profit margin bins
        df['margin_category'] = pd.cut(
            df['profit_margin'],
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=['Loss', 'Low (0-10%)', 'Medium (10-20%)', 'High (20-30%)', 'Very High (>30%)']
        )
        
        margin_distribution = df['margin_category'].value_counts().to_dict()
        
        # Correlation with discount
        discount_bins = pd.qcut(df['discount'], q=5, duplicates='drop')
        discount_profit = df.groupby(discount_bins)['profit_margin'].mean()
        
        return {
            'overall_margin': float(df['profit_margin'].mean()),
            'margin_distribution': {str(k): int(v) for k, v in margin_distribution.items()},
            'category_margins': df.groupby('category')['profit_margin'].mean().to_dict(),
            'discount_impact': {
                'discount_ranges': [str(x) for x in discount_profit.index],
                'avg_profit_margins': discount_profit.tolist()
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting profit analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/seasonal-trends")
async def get_seasonal_trends(db: AsyncSession = Depends(get_db)):
    """Get seasonal sales trends"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'order_date': o.order_date,
            'sales': o.sales,
            'profit': o.profit
        } for o in orders])
        
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['month'] = df['order_date'].dt.month
        df['quarter'] = df['order_date'].dt.quarter
        df['day_of_week'] = df['order_date'].dt.dayofweek
        
        # Monthly trends
        monthly = df.groupby('month').agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        # Quarterly trends
        quarterly = df.groupby('quarter').agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        # Day of week trends (0=Monday, 6=Sunday)
        dow = df.groupby('day_of_week').agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'monthly': {
                'months': monthly['month'].tolist(),
                'sales': monthly['sales'].tolist(),
                'profit': monthly['profit'].tolist()
            },
            'quarterly': {
                'quarters': quarterly['quarter'].tolist(),
                'sales': quarterly['sales'].tolist(),
                'profit': quarterly['profit'].tolist()
            },
            'day_of_week': {
                'days': [dow_names[i] for i in dow['day_of_week']],
                'sales': dow['sales'].tolist(),
                'profit': dow['profit'].tolist()
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting seasonal trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlation-matrix")
async def get_correlation_matrix(db: AsyncSession = Depends(get_db)):
    """Get correlation matrix for numerical features"""
    try:
        result = await db.execute(select(Order))
        orders = result.scalars().all()
        
        if not orders:
            raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame([{
            'sales': o.sales,
            'profit': o.profit,
            'discount': o.discount,
            'quantity': o.order_quantity,
            'price': o.product_price,
            'shipping_days': o.days_for_shipping_real
        } for o in orders])
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        return {
            'features': corr_matrix.columns.tolist(),
            'correlation_matrix': corr_matrix.values.tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting correlation matrix: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
