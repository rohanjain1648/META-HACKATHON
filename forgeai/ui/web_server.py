"""Web Server — FastAPI backend for the ForgeAI dashboard.

Provides real-time observability and control over the agentic pipeline.
"""

import asyncio
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from forgeai.core.activity_logger import LogEntry

app = FastAPI(title="ForgeAI Dashboard")

# In-memory storage for active sessions (simplified)
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# This would be used by the Orchestrator to push updates
async def notify_web_dashboard(event_type: str, data: dict):
    await manager.broadcast({
        "type": event_type,
        "payload": data
    })

def start_web_server(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
