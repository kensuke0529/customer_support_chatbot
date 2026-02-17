"""
Customer Support AI Agent - FastAPI Application
A production-ready AI agent deployed on AWS App Runner with:
- Decoupled health check endpoint (critical for AWS App Runner)
- Lazy loading of heavy AI components
- LangServe integration for /invoke, /batch, /stream endpoints
- Static file serving for the chatbot UI
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

app = FastAPI(
    title="Customer Support AI Agent",
    version="1.0.0",
    description="""
    ## Endpoints
    - `/health` - Health check for AWS App Runner
    - `/chat` - Main chat endpoint with session management
    - `/agent/*` - LangServe endpoints (invoke, batch, stream)
    """,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = project_root / "static"
if static_path.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    except Exception as e:
        print(f"⚠️  Warning: Could not mount static files: {e}")


class ChatRequest(BaseModel):
    """Chat request model with optional session management"""

    user_message: str
    thread_id: Optional[str] = None
    include_history: bool = False


class HealthResponse(BaseModel):

    status: str
    version: str = "1.0.0"
    agent_loaded: bool = False


_agent_module = None
_chat_func = None
_langgraph_app = None


def _load_agent():
    global _agent_module, _chat_func, _langgraph_app
    if _agent_module is None:
        print("🔄 Loading AI agent components...")
        from agent import chat, app as langgraph_app

        _chat_func = chat
        _langgraph_app = langgraph_app
        _agent_module = True
    return _chat_func, _langgraph_app


@app.get("/health")
async def health_check():
    from fastapi import Response

    return Response(status_code=200, media_type="text/plain")


# Root Endpoint (Chatbot UI)
@app.get("/", tags=["UI"])
async def read_root():
    """Serve the chatbot UI or return API info"""
    static_file = project_root / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {
        "message": "Customer Support AI Agent is running",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat",
    }


# Main Chat Endpoint (triggers lazy loading)
@app.post("/chat", tags=["Chat"])
def talk_to_chatbot(request: ChatRequest):

    try:
        chat_func, _ = _load_agent()

        result, thread_id = chat_func(request.user_message, request.thread_id)

        session_info = {
            "user_email": result.get("user_email"),
            "user_name": result.get("user_name"),
            "order_id": result.get("order_id"),
            "contact_info_source": result.get("contact_info_source", "none"),
            "session_id": result.get("session_id"),
            "thread_id": thread_id,
            "response": result.get("response", ""),
        }

        # Optionally include conversation history
        if request.include_history:
            messages = result.get("messages", [])
            history = []
            for msg in messages:
                role = (
                    "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                )
                history.append({"role": role, "content": msg.content})
            session_info["history"] = history
            session_info["message_count"] = len(history)

        return session_info

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in /chat endpoint: {error_details}")
        return {
            "error": "Internal server error",
            "message": str(e),
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "thread_id": request.thread_id or "error",
        }


# Warmup Endpoint (optional - pre-load agent)
@app.post("/warmup", tags=["Health"])
async def warmup():
    """
    Pre-load AI agent components.

    Call this after deployment to warm up the agent before user traffic.
    Useful for reducing latency on first user request.
    """
    try:
        _load_agent()
        return {"status": "warmed_up", "agent_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@app.on_event("startup")
async def setup_langserve():
    """Setup LangServe routes after startup (non-blocking)."""

    pass


@app.get("/agent/status", tags=["Agent"])
async def agent_status():
    """Check if LangServe agent routes are available."""
    return {
        "agent_loaded": _agent_module is not None,
        "message": "Agent loaded"
        if _agent_module
        else "Call /warmup or /chat first to load agent",
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
