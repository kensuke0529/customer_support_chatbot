import sys
from pathlib import Path
from typing import Optional
import traceback

# Add the project root to the path (must be before local imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Import the chat function
from agent import chat  # noqa: E402

app = FastAPI(
    title="Customer Support Agent",
    description="An AI-powered customer support agent that automatically handles customer inquiries with accurate, policy-based responses and intelligent escalation.",
)


@app.get("/")
async def read_root():
    """Serve the chatbot UI"""
    static_file = project_root / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {"message": "Chatbot API is running. Visit /api/health for health check."}


class ChatRequest(BaseModel):
    user_message: str
    thread_id: Optional[str] = None
    include_history: bool = False


@app.post("/api/chat")
async def talk_to_chatbot(request: ChatRequest):
    """Handle chat requests"""
    try:
        result, thread_id = chat(request.user_message, request.thread_id)

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
        print(f"Error in /api/chat endpoint: {error_details}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "thread_id": request.thread_id or "error",
            },
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


# Vercel serverless function handler
# Vercel's @vercel/python automatically detects ASGI apps
# Export the app for Vercel
__all__ = ["app"]
