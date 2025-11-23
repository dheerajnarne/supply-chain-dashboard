
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
import json
from app.services.websocket import manager

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'ping':
                await websocket.send_json({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
