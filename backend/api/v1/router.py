from fastapi import APIRouter
from app.api.v1.endpoints import data, dashboard, forecasts, inventory, routes, websocket, anomalies, auth, analytics

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(forecasts.router, prefix="/forecasts", tags=["forecasts"])
api_router.include_router(inventory.router, prefix="/inventory", tags=["inventory"])
api_router.include_router(routes.router, prefix="/routes", tags=["routes"])
api_router.include_router(anomalies.router, prefix="/anomalies", tags=["anomalies"])
api_router.include_router(websocket.router, tags=["websocket"])
