
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index
from sqlalchemy.sql import func as sql_func
from app.db.base import Base

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, nullable=False, index=True)
    order_date = Column(DateTime(timezone=True), nullable=False, index=True)
    
    customer_id = Column(Integer, nullable=False, index=True)
    customer_country = Column(String(100))
    customer_city = Column(String(100))
    customer_state = Column(String(100))
    customer_segment = Column(String(50))
    
    product_id = Column(Integer, nullable=False, index=True)
    product_name = Column(Text)
    product_category = Column(String(100), index=True)
    product_price = Column(Float)
    
    order_quantity = Column(Integer)
    sales = Column(Float)
    discount = Column(Float)
    profit = Column(Float)
    
    shipping_mode = Column(String(50))
    delivery_status = Column(String(50), index=True)
    days_for_shipping_real = Column(Integer)
    days_for_shipment_scheduled = Column(Integer)
    
    latitude = Column(Float)
    longitude = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=sql_func.now())
    
    __table_args__ = (
        Index('idx_order_date_product', 'order_date', 'product_id'),
        Index('idx_order_date_category', 'order_date', 'product_category'),
    )
