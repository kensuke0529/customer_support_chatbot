from langchain_core.messages import AIMessage
from state import ChatbotInfo
from .db import create_escalation, append_message

def log_escalation(state: ChatbotInfo, reason: str):
    """Logs escalation data to RDS"""
    metadata_dict = {
        "validation_reason": reason,
        "retry_count": state.response_retry_count,
        "context_preview": state.context[:300] if state.context else "",
        "validation_status": state.response_validation,
    }

    try:
        record = create_escalation(
            user_message=state.user_message,
            classification_tag=state.classification_tag,
            response=state.response,
            user_email=state.user_email,
            user_name=state.user_name,
            order_id=state.order_id,
            session_id=state.session_id,
            thread_id=state.thread_id,
            contact_info_source=state.contact_info_source,
            metadata=metadata_dict,
        )
        if record:
            raw_id = record.get("escalation_id") or record.get("id")
            if raw_id:
                print(f"Escalation logged to RDS with ID: {raw_id}")
    except Exception as e:
        print(f"Error logging escalation to RDS: {e}")

def escalation_node(state: ChatbotInfo):
    metadata_dict = {
        "validation_status": state.response_validation,
        "validation_reason": state.response_validation_reason,
        "retry_count": state.response_retry_count,
        "context_preview": state.context[:300] if state.context else "",
    }

    try:
        record = create_escalation(
            user_message=state.user_message,
            classification_tag=state.classification_tag,
            response=state.response,
            user_email=state.user_email,
            user_name=state.user_name,
            order_id=state.order_id,
            session_id=state.session_id,
            thread_id=state.thread_id,
            contact_info_source=state.contact_info_source,
            metadata=metadata_dict,
        )
        escalation_id = None
        if record:
            raw_id = record.get("escalation_id") or record.get("id")
            if raw_id:
                escalation_id = str(raw_id)
                print(f"Escalation created in RDS with ID: {escalation_id}")
    except Exception as e:
        print(f"Error creating escalation in RDS: {e}")
        escalation_id = None

    security_keywords = [
        "hacked", "hack", "fraud", "fraudulent", "unauthorized",
        "security", "breach", "stolen", "identity theft",
    ]
    user_message_lower = state.user_message.lower()
    is_security_emergency = any(keyword in user_message_lower for keyword in security_keywords)
    needs_contact = not state.user_email and not state.user_name

    user_info_section = ""
    if state.user_email or state.user_name or state.order_id:
        user_info_section = "\nUser Information:\n"
        if state.user_email: user_info_section += f"- Email: {state.user_email} ({state.contact_info_source})\n"
        if state.user_name: user_info_section += f"- Name: {state.user_name} ({state.contact_info_source})\n"
        if state.order_id: user_info_section += f"- Order ID: {state.order_id}\n"
        if state.session_id: user_info_section += f"- Session ID: {state.session_id}\n"

    escalation_summary = f"""
Escalation to Human Support

User Query: {state.user_message}
{user_info_section}
Classification:
- Tag: {state.classification_tag}

Attempted Response:
{state.response}

Validation Status: {state.response_validation}
Validation Reason: {state.response_validation_reason}
Retry Attempts: {state.response_retry_count}

Context Used:
{state.context[:300]}...

---
This query has been escalated to a human agent for handling.
"""

    if is_security_emergency:
        response_message = "**Immediate Action Required**\n\nTake steps to secure your account..."
        needs_contact_info = needs_contact
    elif needs_contact:
        response_message = "Thank you. Your query has been escalated. Please provide your email..."
        needs_contact_info = True
    else:
        response_message = "Thank you. Your query has been escalated to our support team."
        needs_contact_info = False

    if state.session_id:
        try:
            append_message(state.session_id, "assistant", response_message)
        except Exception as e:
            print(f"Error storing escalation message: {e}")

    messages = state.messages.copy() if state.messages else []
    if response_message:
        last_msg = messages[-1] if messages else None
        if not (last_msg and isinstance(last_msg, AIMessage) and last_msg.content == response_message):
            messages.append(AIMessage(content=response_message))

    return {
        "response": response_message,
        "messages": messages,
        "escalation_summary": escalation_summary,
        "needs_contact_info": needs_contact_info,
        "escalation_id": escalation_id,
    }
