import json
import re
from langchain_openai import ChatOpenAI
from state import ChatbotInfo
from .base import openai_api_key
from .db import append_message, update_escalation_with_contact_info

def extract_user_info(state: ChatbotInfo):
    if state.session_id:
        try:
            append_message(state.session_id, "user", state.user_message)
        except Exception as e:
            print(f"Error storing user message in DynamoDB: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("session_id is None, skipping DynamoDB storage for user message")

    extraction_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0,
    )

    conversation_text = ""
    if state.messages:
        for msg in state.messages:
            if hasattr(msg, "content"):
                if msg.__class__.__name__ == "HumanMessage":
                    conversation_text += f"User: {msg.content}\n"
                elif msg.__class__.__name__ == "AIMessage":
                    conversation_text += f"Assistant: {msg.content}\n"

    conversation_text += f"User: {state.user_message}\n"

    extraction_prompt = f"""Extract structured information from this customer support conversation.

CONVERSATION:
{conversation_text}

Extract the following information if mentioned:
- Email address
- Customer name 
- Order ID or reference number 

Return ONLY valid JSON:
{{
    "email": "email@example.com" or null,
    "name": "John Doe" or null,
    "order_id": "ORD-12345" or null,
    "account_id": "account123" or null
}}

If information is not found, use null. Be precise - only extract if explicitly mentioned.
"""

    try:
        try:
            response = extraction_llm.invoke(extraction_prompt)
            content = response.content.strip()
        except Exception as e:
            print(f"Error calling extraction LLM: {e}")
            import traceback
            traceback.print_exc()
            return {}

        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in extraction (content: {content[:200]}): {e}")
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            emails = re.findall(email_pattern, state.user_message)
            if emails:
                return {
                    "user_email": emails[0],
                    "email_extracted_this_turn": emails[0],
                    "contact_info_source": "extracted",
                }
            return {}

        updates = {}
        contact_found = False

        if result.get("email") and not state.user_email:
            updates["user_email"] = result["email"]
            updates["email_extracted_this_turn"] = result["email"]
            contact_found = True
        if result.get("name") and not state.user_name:
            updates["user_name"] = result["name"]
        if result.get("order_id") and not state.order_id:
            updates["order_id"] = result["order_id"]

        if contact_found and state.contact_info_source == "none":
            updates["contact_info_source"] = "extracted"

        if updates and (state.session_id or state.thread_id):
            try:
                update_escalation_with_contact_info(
                    session_id=state.session_id,
                    thread_id=state.thread_id,
                    user_email=updates.get("user_email"),
                    user_name=updates.get("user_name"),
                    order_id=updates.get("order_id"),
                    contact_info_source=updates.get("contact_info_source", "extracted"),
                )
            except Exception as e:
                print(f"Error updating escalation with contact info (non-fatal): {e}")
                import traceback
                traceback.print_exc()

        return updates

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in extraction: {e}")
        return {}
    except Exception as e:
        print(f"Extraction error: {e}")
        return {}
