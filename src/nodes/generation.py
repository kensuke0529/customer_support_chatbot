from langchain_openai import ChatOpenAI
from state import ChatbotInfo
from prompts import get_response_prompt
from .base import openai_api_key
from .db import get_history, append_message, is_dynamo_escalations_enabled, get_escalation_by_session

response_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    max_tokens=300,
)

def generate_response(state: ChatbotInfo):
    """Generates a response based on retrieved context, user query, and conversation history."""
    conversation_context = ""
    history_loaded = False

    if state.session_id:
        try:
            past_msgs = get_history(state.session_id, limit=20)
            if past_msgs and len(past_msgs) > 0:
                conversation_context = "\n\nPrevious conversation:\n"
                for m in past_msgs:
                    if m["role"] == "user":
                        conversation_context += f"User: {m['content']}\n"
                    elif m["role"] in ("assistant", "human"):
                        conversation_context += f"Assistant: {m['content']}\n"
                history_loaded = True
                print(f"Loaded {len(past_msgs)} messages from DynamoDB")
        except Exception as e:
            print(f"Error loading history from DynamoDB: {e}")
            import traceback
            traceback.print_exc()

    if not history_loaded and state.messages and len(state.messages) > 1:
        recent_messages = state.messages[-10:]
        conversation_context = "\n\nPrevious conversation:\n"
        for msg in recent_messages:
            if hasattr(msg, "content"):
                if msg.__class__.__name__ == "HumanMessage":
                    role = "User"
                elif msg.__class__.__name__ == "AIMessage":
                    role = "Assistant"
                else:
                    role = "System"
                conversation_context += f"{role}: {msg.content}\n"
        print(f"Using {len(recent_messages)} messages from state.messages")

    user_message_lower = state.user_message.lower()
    if is_dynamo_escalations_enabled() and any(
        keyword in user_message_lower
        for keyword in ["escalation", "escalate", "escalated", "support team", "human agent", "escalation reason"]
    ):
        if state.session_id or state.thread_id:
            try:
                escalation = get_escalation_by_session(session_id=state.session_id, thread_id=state.thread_id)
                if escalation:
                    escalation_reason = escalation.get("user_message", escalation.get("issue_type", ""))
                    classification = escalation.get("classification_tag", escalation.get("issue_type", ""))
                    if escalation_reason:
                        conversation_context += (
                            f"\n\nNote: A previous query was escalated to human support. The escalated query was: '{escalation_reason}'"
                            + (f" (classified as: {classification})" if classification else "")
                            + "."
                        )
            except Exception as e:
                print(f"Error querying escalation info: {e}")
                import traceback
                traceback.print_exc()

    has_contact_info = bool(state.user_email or state.user_name)

    should_ask_for_info = (
        not has_contact_info
        and not state.has_asked_for_contact_info
        and state.contact_ask_count < 1  
        and state.classification_tag in ["billing", "subscription", "account", "returns"]
        and len(state.messages) >= 2  
    )

    just_provided_email = state.email_extracted_this_turn if state.email_extracted_this_turn else None

    context_with_tools = state.context
    if state.tool_results:
        context_with_tools = f"""
{state.context}

[ORDER/PURCHASE]
The following information was retrieved:
{state.tool_results}


Use the above order information to answer the customer's question about their order.
"""
        print("Added tool results to context")

    prompt = get_response_prompt(
        context_with_tools,
        state.user_message,
        conversation_history=conversation_context,
        has_contact_info=has_contact_info,
        should_ask_for_info=should_ask_for_info,
        just_provided_email=just_provided_email,
    )
    response = response_llm.invoke(prompt)
    response_text = response.content

    escalation_phrases = [
        "I'll escalate this to our support team for review.",
        "I'll escalate this to our support team.",
        "I'll escalate",
        "escalate this to our support team",
    ]

    for phrase in escalation_phrases:
        response_text = response_text.replace(phrase, "").strip()
        response_text = response_text.replace(phrase.lower(), "").strip()
        response_text = response_text.replace(phrase.capitalize(), "").strip()

    import re
    response_text = re.sub(r"\.\s*\.", ".", response_text)
    response_text = re.sub(r"\s+", " ", response_text)
    response_text = response_text.strip()

    if not response_text:
        response_text = response.content

    if state.session_id:
        try:
            append_message(state.session_id, "assistant", response_text)
        except Exception as e:
            print(f"Error storing assistant message in DynamoDB: {e}")
            import traceback
            traceback.print_exc()

    result = {"response": response_text}
    if should_ask_for_info:
        result["has_asked_for_contact_info"] = True
        result["contact_ask_count"] = state.contact_ask_count + 1

    return result
