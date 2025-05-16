import asyncio
import logging
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="CompileML API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store reference to the ModelDebugger instance
debugger = None

# API Models
class ModelInfoResponse(BaseModel):
    name: str
    framework: str
    dataset_size: int
    accuracy: float

@app.get("/")
async def root():
    return {"message": "CompileML API is running"}

@app.get("/api/model", response_model=ModelInfoResponse)
async def get_model_info():
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    return {
        "name": debugger.name,
        "framework": debugger.framework,
        "dataset_size": len(debugger.ground_truth) if debugger.ground_truth is not None else 0,
        "accuracy": debugger.analyze()["accuracy"],
    }

@app.get("/api/errors")
async def get_errors():
    if debugger is None:
        raise HTTPException(status_code=404, detail="No model connected")
    
    analysis = debugger.analyze()
    return analysis["error_analysis"]

def start_server(model_debugger, port: int = 8000):
    """Start the FastAPI server with the given ModelDebugger instance."""
    global debugger
    debugger = model_debugger
    
    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    
    # Run the server in a separate thread
    async def run_server():
        await server.serve()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(run_server())
    
    return server