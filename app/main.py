import sys
import traceback
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from agent import chat

app = FastAPI(
    title="Customer Support Agent",
    description="An AI-powered customer support agent that automatically handles customer inquiries with accurate, policy-based responses and intelligent escalation.",
)

# Serve static files (HTML, CSS, JS)
static_path = project_root / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


class ChatRequest(BaseModel):
    user_message: str
    thread_id: Optional[str] = None
    include_history: bool = False  # Optionally return conversation history


@app.get("/")
async def read_root():
    """Serve the chatbot UI"""
    static_file = project_root / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {"message": "Chatbot API is running. Visit /docs for API documentation."}


@app.post("/chat")
def talk_to_chatbot(request: ChatRequest):
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
        print(f"Error in /chat endpoint: {error_details}")
        return {
            "error": "Internal server error",
            "message": str(e),
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "thread_id": request.thread_id or "error",
        }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=9900, reload=True)
