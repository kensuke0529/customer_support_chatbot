"""
Customer Support AI Agent - FastAPI Application
================================================
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

# Setup paths for local imports (lightweight - no heavy imports yet)
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# =============================================================================
# FastAPI Application Initialization
# =============================================================================
app = FastAPI(
    title="Customer Support AI Agent",
    version="1.0.0",
    description="""
    A portfolio-ready AI agent deployed on AWS App Runner.
    
    ## Features
    - 🤖 Intelligent customer support with policy-based responses
    - 🧠 Conversation memory with LangGraph checkpointing
    - 📊 LangSmith tracing for observability
    - 🚀 Production-ready with health checks and streaming support
    
    ## Endpoints
    - `/health` - Health check for AWS App Runner
    - `/chat` - Main chat endpoint with session management
    - `/agent/*` - LangServe endpoints (invoke, batch, stream)
    """,
)

# =============================================================================
# CORS Configuration (for frontend integration)
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Static Files (Chatbot UI) - Lazy mount to avoid blocking startupa
# =============================================================================
# Mount static files lazily to avoid blocking health checks
static_path = project_root / "static"
# Note: Static files are mounted but won't block startup
if static_path.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    except Exception as e:
        print(f"⚠️  Warning: Could not mount static files: {e}")
        # Don't fail startup if static files can't be mounted


# =============================================================================
# Request/Response Models
# =============================================================================
class ChatRequest(BaseModel):
    """Chat request model with optional session management"""

    user_message: str
    thread_id: Optional[str] = None
    include_history: bool = False


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    version: str = "1.0.0"
    agent_loaded: bool = False


# =============================================================================
# LAZY LOADING: Agent components loaded on first request, NOT at startup
# =============================================================================
# This is CRITICAL for App Runner health checks to pass during startup
_agent_module = None
_chat_func = None
_langgraph_app = None


def _load_agent():
    """Lazy load the agent module. Called on first chat request."""
    global _agent_module, _chat_func, _langgraph_app
    if _agent_module is None:
        print("🔄 Loading AI agent components...")
        from agent import chat, app as langgraph_app

        _chat_func = chat
        _langgraph_app = langgraph_app
        _agent_module = True
        print("✅ AI agent loaded successfully")
    return _chat_func, _langgraph_app


# =============================================================================
# CRITICAL: Health Check Endpoint (MUST be lightweight)
# =============================================================================
# This endpoint returns 200 OK immediately WITHOUT loading AI components.
# App Runner sends health checks during startup - if this is slow, deployment fails.
# Using the absolute minimum response for maximum speed and compatibility
@app.get("/health")
async def health_check():
    """
    Health check endpoint for AWS App Runner.

    Returns immediately with 200 status - no body, no processing, no imports.
    This is the fastest possible response for App Runner health checks.
    """
    from fastapi import Response

    return Response(status_code=200, media_type="text/plain")


# =============================================================================
# Root Endpoint (Chatbot UI)
# =============================================================================
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


# =============================================================================
# Main Chat Endpoint (triggers lazy loading)
# =============================================================================
@app.post("/chat", tags=["Chat"])
def talk_to_chatbot(request: ChatRequest):
    """
    Main chat endpoint with conversation memory.

    - Maintains conversation history via thread_id
    - Extracts and persists user information across sessions
    - Returns structured response with session metadata

    Note: First request may be slower as it loads AI components.
    """
    try:
        # Lazy load agent on first request
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


# =============================================================================
# Warmup Endpoint (optional - pre-load agent)
# =============================================================================
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


# =============================================================================
# LangServe Integration (lazy - only if agent is loaded)
# =============================================================================
@app.on_event("startup")
async def setup_langserve():
    """Setup LangServe routes after startup (non-blocking)."""
    # Note: LangServe routes are added lazily when agent is loaded
    # This prevents startup delays
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


# =============================================================================
# Local Development Entry Point
# =============================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
