from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from state import ChatbotInfo
from nodes import (
    classify_intent,
    retrieve_context,
    generate_response,
    response_validation,
    escalation_node,
    update_messages_node,
    extract_user_info,
)
import uuid


# Routing function for validation decision
def validation_router(state: ChatbotInfo):
    """Routes based on validation result"""
    if state.response_validation == "PASS":
        return "end"
    elif state.response_validation == "RETRY" and state.response_retry_count < 3:
        return "retry"
    else:
        return "escalate"


memory = MemorySaver()

workflow = StateGraph(ChatbotInfo)

workflow.add_node("classify", classify_intent)
workflow.add_node("extract_info", extract_user_info)
workflow.add_node("rag", retrieve_context)
workflow.add_node("response", generate_response)
workflow.add_node("response_validation", response_validation)
workflow.add_node("update_messages", update_messages_node)
workflow.add_node("escalate", escalation_node)

workflow.set_entry_point("classify")

workflow.add_edge("classify", "extract_info")
workflow.add_edge("extract_info", "rag")
workflow.add_edge("rag", "response")
workflow.add_edge("response", "response_validation")

workflow.add_conditional_edges(
    "response_validation",
    validation_router,
    {"end": "update_messages", "retry": "response", "escalate": "escalate"},
)
workflow.add_edge("update_messages", END)
workflow.add_edge("escalate", END)

app = workflow.compile(checkpointer=memory)


# Helper function for chat with memory
def chat(user_message: str, thread_id: str = None):
    """
    Chat function with conversation memory using LangGraph checkpointing.

    Args:
        user_message: The user's message
        thread_id: Optional thread ID for conversation continuity. If None, creates a new conversation thread.

    Returns:
        tuple: (result dict, thread_id)
    """
    # Generate thread_id if not provided
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configuration for checkpointing
    config = {"configurable": {"thread_id": thread_id}}

    # Get current state (conversation history) from checkpoint
    current_state = app.get_state(config)
    state_values = current_state.values if current_state.values else {}
    messages = state_values.get("messages", []).copy() if state_values.get("messages") else []

    # Get or generate session_id (persists across conversation)
    session_id = state_values.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    # Preserve extracted user info from previous messages
    user_email = state_values.get("user_email")
    user_name = state_values.get("user_name")
    order_id = state_values.get("order_id")
    contact_info_source = state_values.get("contact_info_source", "none")

    # Add new user message to history
    messages.append(HumanMessage(content=user_message))

    # Prepare initial state - only update fields that need to change
    # This allows LangGraph to properly merge with checkpointed state
    initial_state = {
        "user_message": user_message,
        "messages": messages,  # Include full history
        "thread_id": thread_id,
        "session_id": session_id,
        "user_email": user_email,
        "user_name": user_name,
        "order_id": order_id,
        "contact_info_source": contact_info_source,
        "needs_contact_info": False,
        "classification_tag": "",
        "context": "",
        "response": "",
        "response_validation": "",
        "response_validation_reason": "",
        "response_retry_count": 0,
    }

    # Run the graph with checkpointing
    # Use update_state to properly merge with existing checkpoint
    if current_state.values:
        # Update the state with new message, then invoke
        app.update_state(config, initial_state)
    
    result = app.invoke(initial_state, config=config)

    # Messages are now automatically updated by update_messages_node and saved to checkpoint, so we just need to get them from result
    if "messages" in result:
        messages = result["messages"]
    elif result.get("response"):
        # Fallback: add assistant message if not already in result
        messages.append(AIMessage(content=result["response"]))
        result["messages"] = messages

    return result, thread_id
