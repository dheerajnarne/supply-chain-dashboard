
from pydantic import BaseModel

class DashboardMetrics(BaseModel):
    total_orders: int
    total_sales: float
    total_profit: float
    avg_delivery_days: float
    on_time_delivery_rate: float
    total_products: int
    active_customers: int
