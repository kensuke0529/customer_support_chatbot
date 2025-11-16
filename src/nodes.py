# +++++++++++++++++++++++++++++
# Imports and Setup
# +++++++++++++++++++++++++++++

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from datetime import datetime
from state import ChatbotInfo
from prompts import CLASSIFICATION_PROMPT, get_response_prompt
from supabase_client import (
    add_data_to_supabase,
    update_escalation_with_contact_info,
    add_embeddings_to_supabase,
    search_similar_documents,
    check_embeddings_exist,
    clear_all_embeddings,
    clear_document_embeddings,
)

# Load environment variables
parent_env = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(parent_env)
openai_api_key = os.getenv("OPENAI_API_KEY")

# +++++++++++++++++++++++++++++
# Classification Node
# +++++++++++++++++++++++++++++

classification_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0,
)

chain = CLASSIFICATION_PROMPT | classification_llm


def classify_intent(state: ChatbotInfo):
    """Classifies the user's intent from their message."""
    response = chain.invoke({"input": state.user_message})
    response = response.content
    result = json.loads(response)

    return {
        "classification_tag": result.get("intent", ""),
    }


# +++++++++++++++++++++++++++++
# User Information Extraction Node
# +++++++++++++++++++++++++++++


def extract_user_info(state: ChatbotInfo):
    """
    Extracts user information (email, name, order_id) from conversation messages.
    Uses LLM to intelligently extract structured data from natural language.
    """
    extraction_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0,
    )

    # Build conversation context for extraction
    conversation_text = ""
    if state.messages:
        # Use all messages for better extraction context
        for msg in state.messages:
            if hasattr(msg, "content"):
                if msg.__class__.__name__ == "HumanMessage":
                    conversation_text += f"User: {msg.content}\n"
                elif msg.__class__.__name__ == "AIMessage":
                    conversation_text += f"Assistant: {msg.content}\n"

    # Also include current message
    conversation_text += f"User: {state.user_message}\n"

    extraction_prompt = f"""Extract structured information from this customer support conversation.

CONVERSATION:
{conversation_text}

Extract the following information if mentioned:
- Email address (any email mentioned)
- Customer name (first name, last name, or full name)
- Order ID or reference number (order numbers, transaction IDs, invoice numbers, etc.)
- Account identifier (account number, username, etc.)

Return ONLY valid JSON (no markdown code blocks):
{{
    "email": "email@example.com" or null,
    "name": "John Doe" or null,
    "order_id": "ORD-12345" or null,
    "account_id": "account123" or null
}}

If information is not found, use null. Be precise - only extract if explicitly mentioned.
"""

    try:
        response = extraction_llm.invoke(extraction_prompt)
        content = response.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        result = json.loads(content)

        # Update only if new information is found (don't overwrite existing)
        updates = {}
        contact_found = False

        if result.get("email") and not state.user_email:
            updates["user_email"] = result["email"]
            contact_found = True
        if result.get("name") and not state.user_name:
            updates["user_name"] = result["name"]
        if result.get("order_id") and not state.order_id:
            updates["order_id"] = result["order_id"]

        # Update contact_info_source if we extracted something
        if contact_found and state.contact_info_source == "none":
            updates["contact_info_source"] = "extracted"

        # If we extracted new contact info and have session/thread ID,try to update any existing escalation records in Supabase
        if updates and (state.session_id or state.thread_id):
            update_escalation_with_contact_info(
                session_id=state.session_id or "",
                thread_id=state.thread_id or "",
                user_email=updates.get("user_email"),
                user_name=updates.get("user_name"),
                order_id=updates.get("order_id"),
                contact_info_source=updates.get("contact_info_source", "extracted"),
            )

        return updates

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in extraction: {e}")
        return {}
    except Exception as e:
        print(f"Extraction error: {e}")
        return {}


# +++++++++++++++++++++++++++++
# Context Loader (Utility)
# +++++++++++++++++++++++++++++


def doc_loader(file_path: str, clear_existing: bool = False):
    """
    Loads PDF documents and stores embeddings in Supabase vector database.

    Args:
        file_path: Path to the PDF file
        clear_existing: If True, clears existing embeddings for this document before adding new ones

    Returns:
        Number of chunks processed
    """
    loader = PyPDFLoader(file_path, mode="single")
    docs = loader.load()

    doc_length = len(docs[0].page_content)
    print(f"Document length: {doc_length} characters")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    texts = text_splitter.create_documents([docs[0].page_content])
    print(f"Created {len(texts)} chunks from {file_path}")

    # Get document name from file path
    document_name = os.path.basename(file_path)

    # Check if embeddings already exist for this document
    if clear_existing or not check_embeddings_exist(document_name):
        if clear_existing:
            # Clear existing embeddings for this document
            print(f"Clearing existing embeddings for {document_name}...")
            clear_document_embeddings(document_name)

        # Generate embeddings
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=openai_api_key
        )

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings_list = embeddings_model.embed_documents(
            [text.page_content for text in texts]
        )

        # Prepare data for Supabase
        embeddings_data = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings_list)):
            embeddings_data.append(
                {
                    "content": text.page_content,
                    "embedding": embedding,
                    "document_name": document_name,
                    "chunk_index": i,
                    "metadata": {},
                }
            )

        # Store in Supabase
        if clear_existing:
            # Delete existing embeddings for this document first
            # We'll need to handle this via a direct query or RPC function
            # For now, we'll insert and handle duplicates at the database level
            pass

        add_embeddings_to_supabase(embeddings_data)
        print(f"✅ Stored {len(texts)} chunks in Supabase for {document_name}")
    else:
        print(f"Embeddings already exist for {document_name}, skipping...")

    return len(texts)


# +++++++++++++++++++++++++++++
# Context Retrieval Node
# +++++++++++++++++++++++++++++


def retrieve_context(state: ChatbotInfo):
    """Retrieves relevant context from Supabase vector database based on user query."""
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=openai_api_key
    )

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    try:
        # Check if embeddings exist in Supabase
        if not check_embeddings_exist():
            # Load documents if embeddings don't exist
            print("No embeddings found in Supabase. Loading documents...")
            docs = [
                os.path.join(project_root, "document", "Account Management Policy.pdf"),
                os.path.join(project_root, "document", "Billing & Payment.pdf"),
                os.path.join(project_root, "document", "Contact Information.pdf"),
                os.path.join(project_root, "document", "Customer Support Policy.pdf"),
                os.path.join(
                    project_root, "document", "Shipping & Delivery Policy.pdf"
                ),
                os.path.join(
                    project_root, "document", "Subscription Management Policy.pdf"
                ),
            ]

            loaded_count = 0
            for doc in docs:
                if os.path.exists(doc):
                    doc_loader(doc, clear_existing=False)
                    loaded_count += 1
                else:
                    print(f"Warning: Document not found: {doc}")

            if loaded_count == 0:
                print("Warning: No documents loaded. Returning empty context.")
                return {"context": ""}

            print(f"✅ Loaded {loaded_count} documents into Supabase")

        # Generate embedding for the query
        query_text = f"Question Category: {state.classification_tag} | User Question: {state.user_message}"
        query_embedding = embeddings_model.embed_query(query_text)

        # Search for similar documents
        results = search_similar_documents(
            query_embedding=query_embedding,
            match_threshold=0.7,  # Minimum similarity threshold
            match_count=3,  # Number of results to return
        )

        if results:
            context = "\n\n".join([result["content"] for result in results])
            print(f"✅ Retrieved {len(results)} relevant chunks from Supabase")
            return {"context": context}

        print("⚠️  No similar documents found in Supabase")
        return {"context": ""}

    except Exception as e:
        print(f"Error in retrieve_context: {e}")
        import traceback

        traceback.print_exc()
        return {"context": ""}


# +++++++++++++++++++++++++++++
# Response Generation Node
# +++++++++++++++++++++++++++++


def generate_response(state: ChatbotInfo):
    """Generates a response based on retrieved context, user query, and conversation history."""
    response_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.0,
        max_tokens=300,
    )

    # Build conversation context from message history
    conversation_context = ""
    if state.messages and len(state.messages) > 1:
        # Get last 6 messages (3 user-assistant pairs) for context to avoid token bloat
        recent_messages = state.messages[-6:]
        conversation_context = "\n\nPrevious conversation:\n"
        for msg in recent_messages:
            if hasattr(msg, "content"):
                # Determine role based on message type
                if msg.__class__.__name__ == "HumanMessage":
                    role = "User"
                elif msg.__class__.__name__ == "AIMessage":
                    role = "Assistant"
                else:
                    role = "System"
                conversation_context += f"{role}: {msg.content}\n"

    # Check if we have contact info
    has_contact_info = bool(state.user_email or state.user_name)

    # Determine if we should ask for contact info
    # Ask if: no contact info AND (complex issue OR billing/subscription issue OR user seems to need follow-up)
    should_ask_for_info = (
        not has_contact_info
        and state.classification_tag
        in ["billing", "subscription", "account", "returns"]
        and len(state.messages) >= 2  # After at least one exchange
    )

    prompt = get_response_prompt(
        state.context,
        state.user_message,
        conversation_history=conversation_context,
        has_contact_info=has_contact_info,
        should_ask_for_info=should_ask_for_info,
    )
    response = response_llm.invoke(prompt)

    # Post-process: Remove any escalation language that might have slipped through
    response_text = response.content

    # Remove common escalation phrases
    escalation_phrases = [
        "I'll escalate this to our support team for review.",
        "I don't see that in our current policy. I'll escalate this to our support team for review.",
        "I'll escalate this to our support team.",
        "I'll escalate this to the support team.",
        "I'll escalate this to a human agent.",
        "I'll escalate this to our support team for review",
        "I'll escalate this to our support team",
        "I'll escalate this to the support team",
        "I'll escalate this to a human agent",
        "I'll escalate this",
        "I'll escalate",
        "escalate this to our support team",
        "escalate this to the support team",
        "escalate to our support team",
        "escalate to the support team",
        "escalate to a human agent",
    ]

    for phrase in escalation_phrases:
        # Remove the phrase and any trailing punctuation/whitespace
        response_text = response_text.replace(phrase, "").strip()
        # Also handle case variations
        response_text = response_text.replace(phrase.lower(), "").strip()
        response_text = response_text.replace(phrase.capitalize(), "").strip()

    # Clean up any double spaces or periods that might result
    import re

    response_text = re.sub(r"\.\s*\.", ".", response_text)  # Remove double periods
    response_text = re.sub(r"\s+", " ", response_text)  # Remove extra spaces
    response_text = response_text.strip()

    # If response is empty after cleaning, use original (shouldn't happen, but safety net)
    if not response_text:
        response_text = response.content

    return {"response": response_text}


# +++++++++++++++++++++++++++++
# Utility: Log Escalation
# +++++++++++++++++++++++++++++


def log_escalation(state: ChatbotInfo, reason: str):
    """Logs escalation data to Supabase"""

    escalation_data = {
        "timestamp": datetime.now().isoformat(),
        "user_message": state.user_message,
        "classification_tag": state.classification_tag,
        "response": state.response,
        "validation_reason": reason,
        "retry_count": state.response_retry_count,
        "context_preview": state.context[:300] if state.context else "",
        # User information (extracted or provided)
        "user_email": state.user_email,
        "user_name": state.user_name,
        "order_id": state.order_id,
        "session_id": state.session_id,
        "thread_id": state.thread_id,
        "contact_info_source": state.contact_info_source,  # "extracted", "provided", "none"
    }

    add_data_to_supabase(escalation_data)
    print(f"Escalation data saved to Supabase: {escalation_data}")


# +++++++++++++++++++++++++++++
# Response Validation Node
# +++++++++++++++++++++++++++++


def response_validation(state: ChatbotInfo):
    """
    Validates the generated response for quality, accuracy, and completeness.
    Returns PASS, RETRY, or FAIL status.
    """
    validation_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0,
    )

    validation_prompt = f"""You are validating a customer support response. Be lenient and practical.

User Question: {state.user_message}
Generated Response: {state.response}
Context Available: {state.context[:300]}...

Respond in JSON format (NO markdown code blocks, just raw JSON):
{{
    "status": "PASS" | "RETRY" | "FAIL",
    "reason": "Brief explanation"
}}

CRITICAL: CHECK FOR SECURITY EMERGENCIES FIRST!

FAIL IMMEDIATELY if the user question mentions ANY of these:
- Account hacked / hacked account / account is hacked
- Fraud / fraudulent charges / unauthorized charges
- Credit card used without permission
- Unauthorized access / security breach
- Identity theft / stolen account
- Someone else using my account

If user mentions security emergency → ALWAYS return FAIL regardless of response quality.

OTHER RULES:

PASS if:
- Response attempts to answer the question (even if it mentions escalation or says policy info is limited)
- Response is professional enough
- No security emergency mentioned
- Response provides helpful information or guidance, even if partial

RETRY if:
- Response is not friendly or helpful 
- Response is offensive to anyone
- Response is completely unhelpful or nonsensical

FAIL if:
- Security emergency detected (see above)
- Response would harm the customer
- Response is completely empty or contains only errors

IMPORTANT: If a response mentions "escalate" or "I'll escalate", this is still a valid response attempt and should PASS validation. The system will handle actual escalation separately. Only fail if the response is truly unhelpful, offensive, or harmful.

Default to PASS for normal questions. But ALWAYS FAIL for security emergencies.
"""

    try:
        response = validation_llm.invoke(validation_prompt)
        content = response.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            # Remove opening ```json or ``` and closing ```
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        # Parse JSON
        result = json.loads(content)

        status = result.get("status", "FAIL")
        reason = result.get("reason", "Unknown validation error")

        new_retry_count = state.response_retry_count
        if status == "RETRY":
            new_retry_count += 1

        # Log escalation if security emergency detected
        if status == "FAIL":
            # Check if it's a security emergency based on reason or user message
            security_keywords = [
                "hacked",
                "fraud",
                "unauthorized",
                "security",
                "breach",
                "stolen",
            ]
            user_message_lower = state.user_message.lower()
            reason_lower = reason.lower()

            is_security_emergency = any(
                keyword in user_message_lower or keyword in reason_lower
                for keyword in security_keywords
            )

            if is_security_emergency:
                log_escalation(state, reason)

        return {
            "response_validation": status,
            "response_validation_reason": reason,
            "response_retry_count": new_retry_count,
        }

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in validation: {e}")
        print(f"Raw response content: {response.content[:200]}")
        return {
            "response_validation": "FAIL",
            "response_validation_reason": f"Invalid JSON response from validation: {str(e)}",
            "response_retry_count": state.response_retry_count,
        }
    except Exception as e:
        print(f"Validation error: {e}")
        return {
            "response_validation": "FAIL",
            "response_validation_reason": f"Validation process error: {str(e)}",
            "response_retry_count": state.response_retry_count,
        }


# +++++++++++++++++++++++++++++
# Update Messages Node (for checkpointing)
# +++++++++++++++++++++++++++++


def update_messages_node(state: ChatbotInfo):
    """
    Updates the messages list with the assistant response for checkpointing.
    This ensures conversation history is properly saved.
    """
    from langchain_core.messages import AIMessage

    messages = state.messages.copy() if state.messages else []

    # Add assistant response if it exists and hasn't been added yet
    if state.response and messages:
        # Check if the last message is already the assistant response
        last_msg = messages[-1] if messages else None
        if not (
            last_msg
            and isinstance(last_msg, AIMessage)
            and last_msg.content == state.response
        ):
            messages.append(AIMessage(content=state.response))

    return {"messages": messages}


# +++++++++++++++++++++++++++++
# Escalation Node
# +++++++++++++++++++++++++++++


def escalation_node(state: ChatbotInfo):
    """
    Handles escalation to human support.
    Prepares a comprehensive summary for the human agent.
    If no contact info found, prompts user for email.
    """
    # Check if we need to ask for contact information
    needs_contact = not state.user_email and not state.user_name

    # Build escalation message with user info
    user_info_section = ""
    if state.user_email or state.user_name or state.order_id:
        user_info_section = "\nUser Information:\n"
        if state.user_email:
            user_info_section += (
                f"- Email: {state.user_email} ({state.contact_info_source})\n"
            )
        if state.user_name:
            user_info_section += (
                f"- Name: {state.user_name} ({state.contact_info_source})\n"
            )
        if state.order_id:
            user_info_section += f"- Order ID: {state.order_id}\n"
        if state.session_id:
            user_info_section += f"- Session ID: {state.session_id}\n"

    escalation_message = f"""
🔴 ESCALATION TO HUMAN SUPPORT

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
Please review the context and provide appropriate assistance to the customer.
"""

    # Response message - ask for email if no contact info
    if needs_contact:
        response_message = (
            "Thank you for your patience. Your query has been escalated to our support team. "
            "To ensure we can follow up with you, could you please provide your email address? "
            "A human agent will assist you shortly."
        )
        needs_contact_info = True
    else:
        response_message = (
            "Thank you for your patience. Your query has been escalated to our support team. "
            "A human agent will assist you shortly."
        )
        needs_contact_info = False

    return {
        "response": response_message,
        "escalation_summary": escalation_message,
        "needs_contact_info": needs_contact_info,
    }


# +++++++++++++++++++++++++++++
# Utility: Load Documents
# +++++++++++++++++++++++++++++
# Uncomment and run this to load your documents:
"""
if __name__ == "__main__":
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    docs = [
        os.path.join(project_root, "document", "Account Management Policy.pdf"),
        os.path.join(project_root, "document", "Billing & Payment.pdf"),
        os.path.join(project_root, "document", "Contact Information.pdf"),
        os.path.join(project_root, "document", "Customer Support Policy.pdf"),
        os.path.join(project_root, "document", "Shipping & Delivery Policy.pdf"),
        os.path.join(project_root, "document", "Subscription Management Policy.pdf"),
    ]

    vector_store = None
    for doc in docs:
        vector_store = doc_loader(doc, vector_store)

    print("All documents loaded successfully!")
"""
