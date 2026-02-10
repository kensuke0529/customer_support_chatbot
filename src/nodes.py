import os
import sys
from pathlib import Path
import numpy as np
import boto3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json
import re

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from state import ChatbotInfo
from prompts import CLASSIFICATION_PROMPT, get_response_prompt

_tools_loaded = False
_tools = None


def doc_loader(pdf_path: str, clear_existing: bool = False, return_chunks: bool = False):

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = text_splitter.split_documents(documents)
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    if return_chunks:
        return chunk_texts
    return len(chunk_texts)


_db_functions_loaded = False
_append_message = None
_get_history = None
_create_escalation = None
_update_escalation_with_contact_info = None
_get_escalation_by_session = None
USE_DYNAMO_ESCALATIONS = False


def _load_db_functions():
    """Lazy load database functions to avoid blocking container startup."""
    global _db_functions_loaded, _append_message, _get_history
    global \
        _create_escalation, \
        _update_escalation_with_contact_info, \
        _get_escalation_by_session
    global USE_DYNAMO_ESCALATIONS

    if _db_functions_loaded:
        return

    try:
        from backend.db.chat_memory_dynamo import append_message as _am, get_history as _gh

        _append_message = _am
        _get_history = _gh
        print("DynamoDB chat history enabled")
    except Exception as e:
        print(f"DynamoDB chat history import failed: {e}")
        _append_message = lambda *args, **kwargs: None
        _get_history = lambda *args, **kwargs: []

    try:
        from backend.db.escalations_dynamo import (
            create_escalation as _ce,
            update_escalation_with_contact_info as _ueci,
            get_escalation_by_session as _gebs,
        )

        _create_escalation = _ce
        _update_escalation_with_contact_info = _ueci
        _get_escalation_by_session = _gebs
        USE_DYNAMO_ESCALATIONS = True
        print("DynamoDB escalations enabled")
    except Exception as e:
        print(f"DynamoDB escalations import failed: {e}")
        print("Escalations will be logged to console only")
        USE_DYNAMO_ESCALATIONS = False
        _create_escalation = lambda **kwargs: {"escalation_id": "console-only"}
        _update_escalation_with_contact_info = lambda **kwargs: None
        _get_escalation_by_session = lambda **kwargs: None

    _db_functions_loaded = True


def append_message(*args, **kwargs):
    _load_db_functions()
    return _append_message(*args, **kwargs)


def get_history(*args, **kwargs):
    _load_db_functions()
    return _get_history(*args, **kwargs)


def create_escalation(**kwargs):
    _load_db_functions()
    return _create_escalation(**kwargs)


def update_escalation_with_contact_info(**kwargs):
    _load_db_functions()
    return _update_escalation_with_contact_info(**kwargs)


def get_escalation_by_session(**kwargs):
    _load_db_functions()
    return _get_escalation_by_session(**kwargs)


def is_dynamo_escalations_enabled():
    _load_db_functions()
    return USE_DYNAMO_ESCALATIONS


parent_env = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(parent_env)
load_dotenv()  # Also try current directory
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print(
        "WARNING: OPENAI_API_KEY not set. Chat functionality will fail at runtime."
    )


_classification_llm = None
_classification_chain = None


def get_classification_chain():
    """Lazy initialization of classification LLM to avoid startup delays."""
    global _classification_llm, _classification_chain
    if _classification_llm is None:
        _classification_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0,
        )
        _classification_chain = CLASSIFICATION_PROMPT | _classification_llm
    return _classification_chain


def classify_intent(state: ChatbotInfo):
    """Classifies the user's intent from their message."""
    chain = get_classification_chain()
    response = chain.invoke({"input": state.user_message})
    response = response.content
    result = json.loads(response)

    return {
        "classification_tag": result.get("intent", ""),
    }




def classify_and_extract_parallel(state: ChatbotInfo):

    from concurrent.futures import ThreadPoolExecutor

    results = {}

    def run_classification():
        try:
            return classify_intent(state)
        except Exception as e:
            print(f"⚠️  Classification error: {e}")
            return {"classification_tag": "general"}

    def run_extraction():
        try:
            return extract_user_info(state)
        except Exception as e:
            print(f"⚠️  Extraction error: {e}")
            return {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        classify_future = executor.submit(run_classification)
        extract_future = executor.submit(run_extraction)

        # Gather results
        try:
            classify_result = classify_future.result(timeout=30)
            results.update(classify_result)
        except Exception as e:
            print(f"Classification timeout/error: {e}")
            results["classification_tag"] = "general"

        try:
            extract_result = extract_future.result(timeout=30)
            results.update(extract_result)
        except Exception as e:
            print(f"Extraction timeout/error: {e}")

    print(
        f"Parallel classify+extract completed: tag={results.get('classification_tag')}"
    )
    return results




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
        try:
            response = extraction_llm.invoke(extraction_prompt)
            content = response.content.strip()
        except Exception as e:
            print(f"⚠️  Error calling extraction LLM: {e}")
            import traceback

            traceback.print_exc()
            return {}  # Return empty updates if LLM call fails

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
            print(
                f"⚠️  JSON parsing error in extraction (content: {content[:200]}): {e}"
            )
            import re

            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            emails = re.findall(email_pattern, state.user_message)
            if emails:
                return {
                    "user_email": emails[0],
                    "email_extracted_this_turn": emails[0],  # Track for acknowledgment
                    "contact_info_source": "extracted",
                }
            return {}

        updates = {}
        contact_found = False

        if result.get("email") and not state.user_email:
            updates["user_email"] = result["email"]
            updates["email_extracted_this_turn"] = result[
                "email"
            ]  # Track for acknowledgment
            contact_found = True
        if result.get("name") and not state.user_name:
            updates["user_name"] = result["name"]
        if result.get("order_id") and not state.order_id:
            updates["order_id"] = result["order_id"]

        if contact_found and state.contact_info_source == "none":
            updates["contact_info_source"] = "extracted"

        # If we extracted new contact info and have session/thread ID, try to update any existing escalation records in RDS
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
                # Don't fail the entire request if escalation update fails
                print(
                    f"⚠️  Error updating escalation with contact info (non-fatal): {e}"
                )
                import traceback

                traceback.print_exc()

        return updates

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in extraction: {e}")
        return {}
    except Exception as e:
        print(f"Extraction error: {e}")
        return {}




def _load_tools():
    global _tools_loaded, _tools
    if _tools_loaded:
        return _tools

    try:
        from tools import get_tools

        _tools = get_tools()
        _tools_loaded = True
        print(f"✅ Loaded {len(_tools)} tools")
        return _tools
    except Exception as e:
        print(f"⚠️ Could not load tools: {e}")
        _tools_loaded = True
        _tools = []
        return []


def _should_use_tools(state: ChatbotInfo) -> bool:

    order_keywords = [
        "order",
        "purchase",
        "bought",
        "tracking",
        "shipment",
        "delivery",
        "where is my",
        "order status",
        "order history",
        "my orders",
        "recent orders",
        "ord-",
    ]
    message_lower = state.user_message.lower()
    return any(keyword in message_lower for keyword in order_keywords)


def _extract_order_id_from_message(message: str) -> str | None:
    """Extract order ID from message if present."""
    patterns = [
        r"ORD-\d+",
        r"ord-\d+",
        r"order\s*#?\s*(\d{5,})",
        r"#(\d{5,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            result = match.group(0)
            if not result.upper().startswith("ORD-"):
                num_match = re.search(r"\d+", result)
                if num_match:
                    return f"ORD-{num_match.group(0)}"
            return result.upper()
    return None


def execute_tools(state: ChatbotInfo):
    """
    Execute relevant tools based on user query.
    This node checks if the user is asking about orders and calls the appropriate tool.
    """
    if not _should_use_tools(state):
        print("⏭️ Skipping tools - not an order-related query")
        return {"tool_results": None, "tool_calls_made": []}

    tools = _load_tools()
    if not tools:
        print("⚠️ No tools available")
        return {"tool_results": None, "tool_calls_made": []}

    tool_results = []
    tool_calls = []

    tool_map = {tool.name: tool for tool in tools}

    order_id = _extract_order_id_from_message(state.user_message)

    if not order_id and state.order_id:
        order_id = state.order_id

    message_lower = state.user_message.lower()

    try:
        if order_id:
            if (
                "status" in message_lower
                or "where" in message_lower
                or "track" in message_lower
            ):
                if "check_order_status" in tool_map:
                    result = tool_map["check_order_status"].invoke(order_id)
                    tool_results.append(result)
                    tool_calls.append("check_order_status")
                    print(f"✅ Called check_order_status for {order_id}")
            else:
                if "get_order_details" in tool_map:
                    result = tool_map["get_order_details"].invoke(order_id)
                    tool_results.append(result)
                    tool_calls.append("get_order_details")
                    print(f"✅ Called get_order_details for {order_id}")

        elif state.user_email:
            if "lookup_customer_orders" in tool_map:
                result = tool_map["lookup_customer_orders"].invoke(state.user_email)
                tool_results.append(result)
                tool_calls.append("lookup_customer_orders")
                print(f"✅ Called lookup_customer_orders for {state.user_email}")

        else:
            tool_results.append(
                "To look up your order information, I'll need either your email address or order number (like ORD-123456)."
            )
            tool_calls.append("info_needed")
            print("ℹ️ Need email or order ID to look up orders")

    except Exception as e:
        print(f"❌ Tool execution error: {e}")
        import traceback

        traceback.print_exc()
        tool_results.append(
            "Sorry, I encountered an error looking up that information."
        )

    combined_results = "\n\n".join(tool_results) if tool_results else None

    return {
        "tool_results": combined_results,
        "tool_calls_made": tool_calls,
    }



S3_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_VECTOR_BUCKET = os.getenv("VECTOR_BUCKET", "s3-vector-chatbot-policy-docs")
S3_VECTOR_INDEX = os.getenv("VECTOR_INDEX", "my-s3-vector-index")

_s3v_client = None


def get_s3v_client():
    """Lazy initialization of S3 Vectors client to avoid startup hangs."""
    global _s3v_client
    if _s3v_client is None:
        _s3v_client = boto3.client("s3vectors", region_name=S3_REGION)
    return _s3v_client


def retrieve_context(state: ChatbotInfo):
    try:
        return retrieve_context_rag(state)
    except Exception as e:
        error_msg = f"❌ RAG FAILED: S3 Vectors retrieval failed: {e}"
        print(error_msg)
        import traceback

        traceback.print_exc()
        print(f"   S3_REGION: {S3_REGION}")
        print(f"   S3_VECTOR_BUCKET: {S3_VECTOR_BUCKET}")
        print(f"   S3_VECTOR_INDEX: {S3_VECTOR_INDEX}")
        return {"context": ""}


def retrieve_context_rag(state: ChatbotInfo, top_k: int = 5):

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=openai_api_key
    )

    query_text = f"{state.classification_tag} question: {state.user_message}"

    query_embedding = embeddings_model.embed_query(query_text)
    query_vec = np.array(query_embedding, dtype=np.float32).tolist()

    try:
        resp = get_s3v_client().query_vectors(
            vectorBucketName=S3_VECTOR_BUCKET,
            indexName=S3_VECTOR_INDEX,
            queryVector={"float32": query_vec},
            topK=top_k,
            returnMetadata=True,  # Get associated metadata (doc name, chunk text, etc)
        )

        hits = resp.get("vectors", [])
        if not hits:
            print("⚠️  No similar vectors found via S3 Vectors")
            return {"context": ""}

        # 3. Extract text from metadata for each hit
        context_chunks = []
        for hit in hits:
            meta = hit.get("metadata", {})
            # Text is stored in metadata when we store vectors
            text = meta.get("text")
            if text:
                context_chunks.append(text)
            else:
                # Fallback: try to reconstruct from source_doc and chunk_index if text not in metadata
                doc_name = meta.get("source_doc")
                chunk_idx = meta.get("chunk_index")
                if doc_name and chunk_idx is not None:
                    print(
                        f"⚠️  Text not found in metadata for {doc_name} chunk {chunk_idx}"
                    )

        if context_chunks:
            context = "\n\n".join(context_chunks)
            print(f"✅ Retrieved {len(context_chunks)} chunks from S3 Vectors")
            return {"context": context}
        else:
            print("⚠️  No text content found in retrieved vectors")
            return {"context": ""}

    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"❌ RAG ERROR: Failed to query S3 Vectors ({error_type}): {e}"
        print(error_msg)
        import traceback

        traceback.print_exc()
        # Log helpful debugging info
        print(f"   Query: {query_text[:100]}...")
        print(
            f"   Bucket: {S3_VECTOR_BUCKET}, Index: {S3_VECTOR_INDEX}, Region: {S3_REGION}"
        )
        # Check if it's a permissions issue
        if "AccessDenied" in str(e) or "UnauthorizedOperation" in str(e):
            print(
                "   ⚠️  This looks like an IAM permissions issue. Check App Runner instance role permissions."
            )
        elif "NoSuchBucket" in str(e) or "NoSuchIndex" in str(e):
            print(
                "   ⚠️  The S3 Vectors bucket or index doesn't exist. Run load_documents.py to create it."
            )
        return {"context": ""}




def generate_response(state: ChatbotInfo):
    """Generates a response based on retrieved context, user query, and conversation history."""
    response_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.0,
        max_tokens=300,
    )

    conversation_context = ""
    history_loaded = False

    if state.session_id:
        try:
            # Get last 20 messages from DynamoDB (more comprehensive history)
            past_msgs = get_history(state.session_id, limit=20)

            if past_msgs and len(past_msgs) > 0:
                conversation_context = "\n\nPrevious conversation:\n"
                for m in past_msgs:
                    if m["role"] == "user":
                        conversation_context += f"User: {m['content']}\n"
                    elif m["role"] in ("assistant", "human"):
                        conversation_context += f"Assistant: {m['content']}\n"
                history_loaded = True
                print(f"✅ Loaded {len(past_msgs)} messages from DynamoDB")
        except Exception as e:
            print(f"⚠️  Error loading history from DynamoDB: {e}")
            import traceback

            traceback.print_exc()

    if not history_loaded and state.messages and len(state.messages) > 1:
        recent_messages = state.messages[-10:]  # Get more messages from state
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
        print(f"✅ Using {len(recent_messages)} messages from state.messages")

    user_message_lower = state.user_message.lower()
    if is_dynamo_escalations_enabled() and any(
        keyword in user_message_lower
        for keyword in [
            "escalation",
            "escalate",
            "escalated",
            "support team",
            "human agent",
            "escalation reason",
        ]
    ):
        # Try to get escalation information from DynamoDB
        if state.session_id or state.thread_id:
            try:
                escalation = get_escalation_by_session(
                    session_id=state.session_id, thread_id=state.thread_id
                )

                if escalation:
                    escalation_reason = escalation.get(
                        "user_message", escalation.get("issue_type", "")
                    )
                    classification = escalation.get(
                        "classification_tag",
                        escalation.get("issue_type", ""),
                    )
                    if escalation_reason:
                        conversation_context += (
                            f"\n\nNote: A previous query was escalated to human support. The escalated query was: '{escalation_reason}'"
                            + (
                                f" (classified as: {classification})"
                                if classification
                                else ""
                            )
                            + "."
                        )
            except Exception as e:
                print(f"⚠️  Error querying escalation info: {e}")
                import traceback

                traceback.print_exc()

    has_contact_info = bool(state.user_email or state.user_name)

    should_ask_for_info = (
        not has_contact_info
        and not state.has_asked_for_contact_info
        and state.contact_ask_count < 1  
        and state.classification_tag
        in ["billing", "subscription", "account", "returns"]
        and len(state.messages) >= 2  
    )

    # Check if email was just provided this turn (for acknowledgment)
    just_provided_email = (
        state.email_extracted_this_turn if state.email_extracted_this_turn else None
    )

    context_with_tools = state.context
    if state.tool_results:
        context_with_tools = f"""
{state.context}

=== ORDER/PURCHASE INFORMATION ===
The following information was retrieved from our system:
{state.tool_results}
=== END ORDER INFO ===

Use the above order information to answer the customer's question about their order.
"""
        print("✅ Added tool results to context")

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
        response_text = response_text.replace(phrase, "").strip()
        response_text = response_text.replace(phrase.lower(), "").strip()
        response_text = response_text.replace(phrase.capitalize(), "").strip()

    import re

    response_text = re.sub(r"\.\s*\.", ".", response_text)  # Remove double periods
    response_text = re.sub(r"\s+", " ", response_text)  # Remove extra spaces
    response_text = response_text.strip()

    if not response_text:
        response_text = response.content

    if state.session_id:
        try:
            append_message(state.session_id, "assistant", response_text)
        except Exception as e:
            print(f"⚠️  Error storing assistant message in DynamoDB: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠️  session_id is None, skipping DynamoDB storage for assistant message")

    result = {"response": response_text}

    if should_ask_for_info:
        result["has_asked_for_contact_info"] = True
        result["contact_ask_count"] = state.contact_ask_count + 1

    return result




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

        escalation_id = None
        if record:
            raw_id = record.get("escalation_id") or record.get("id")
            if raw_id:
                escalation_id = str(raw_id)  # Convert to string for consistency
                print(f"✅ Escalation logged to RDS with ID: {escalation_id}")
    except Exception as e:
        print(f"⚠️  Error logging escalation to RDS: {e}")




def _fast_security_check(user_message: str) -> bool:
    """
    Rule-based check for security emergencies. Returns True if security issue detected.
    This is faster than LLM and catches critical cases immediately.
    """
    security_keywords = [
        "hacked",
        "hack",
        "fraud",
        "fraudulent",
        "unauthorized",
        "security breach",
        "stolen",
        "identity theft",
        "someone else",
        "not me",
        "didn't authorize",
        "suspicious activity",
        "compromised",
    ]
    message_lower = user_message.lower()
    return any(keyword in message_lower for keyword in security_keywords)


def _fast_response_check(response: str) -> tuple[bool, str]:
    if not response or len(response.strip()) < 10:
        return False, "Response is empty or too short"

    bad_patterns = [
        "i don't know",
        "i cannot help",
        "error occurred",
        "something went wrong",
        "undefined",
        "null",
    ]
    response_lower = response.lower()
    for pattern in bad_patterns:
        if pattern in response_lower and len(response) < 50:
            return False, f"Response contains problematic pattern: {pattern}"

    return True, "Basic checks passed"


def response_validation(state: ChatbotInfo):
    if _fast_security_check(state.user_message):
        print("⚠️  Security emergency detected via fast check - escalating")
        log_escalation(state, "Security emergency detected (fast path)")
        return {
            "response_validation": "FAIL",
            "response_validation_reason": "Security emergency detected",
            "response_retry_count": state.response_retry_count,
        }

    is_valid, reason = _fast_response_check(state.response)
    if not is_valid:
        print(f"⚠️  Fast validation failed: {reason}")
        return {
            "response_validation": "RETRY",
            "response_validation_reason": reason,
            "response_retry_count": state.response_retry_count + 1,
        }

    simple_categories = ["general"]
    if state.classification_tag in simple_categories and len(state.response) > 30:
        print("✅ Fast validation PASS for simple query (skipping LLM)")
        return {
            "response_validation": "PASS",
            "response_validation_reason": "Fast path: simple query with adequate response",
            "response_retry_count": state.response_retry_count,
        }

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

        if content.startswith("```"):
            # Remove opening ```json or ``` and closing ```
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        result = json.loads(content)

        status = result.get("status", "FAIL")
        reason = result.get("reason", "Unknown validation error")

        new_retry_count = state.response_retry_count
        if status == "RETRY":
            new_retry_count += 1

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




def update_messages_node(state: ChatbotInfo):

    from langchain_core.messages import AIMessage

    messages = state.messages.copy() if state.messages else []

    if state.response and messages:
        last_msg = messages[-1] if messages else None
        if not (
            last_msg
            and isinstance(last_msg, AIMessage)
            and last_msg.content == state.response
        ):
            messages.append(AIMessage(content=state.response))

    return {"messages": messages}




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

        # Extract escalation_id from the returned record
        escalation_id = None
        if record:
            # The record is a RealDictRow, so we can access it like a dict
            raw_id = record.get("escalation_id") or record.get("id")
            if raw_id:
                escalation_id = str(raw_id)  # Convert to string for state compatibility
                print(f"✅ Escalation created in RDS with ID: {escalation_id}")
    except Exception as e:
        print(f"⚠️  Error creating escalation in RDS: {e}")
        escalation_id = None

    security_keywords = [
        "hacked",
        "hack",
        "fraud",
        "fraudulent",
        "unauthorized",
        "security",
        "breach",
        "stolen",
        "identity theft",
    ]
    user_message_lower = state.user_message.lower()
    is_security_emergency = any(
        keyword in user_message_lower for keyword in security_keywords
    )

    needs_contact = not state.user_email and not state.user_name

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

    if is_security_emergency:
        response_message = (
            "⚠️ **Immediate Action Required**\n\n"
            "Please take these steps right away to secure your account:\n\n"
            "1. **Report this immediately** to security@company.com\n"
            "2. **Change your password** immediately: Settings > Account > Password\n"
            "3. **Enable Two-Factor Authentication (2FA)** for added security: Settings > Security > 2FA\n"
            "4. **Review recent account activity** for any unauthorized actions\n\n"
        )
        if needs_contact:
            response_message += (
                "We've escalated this to our security team. To ensure we can follow up with you, "
                "could you please provide your email address? A security specialist will contact you shortly."
            )
        else:
            response_message += (
                "We've escalated this to our security team. A security specialist will review your account "
                "and contact you shortly to provide additional assistance."
            )
        needs_contact_info = needs_contact
    elif needs_contact:
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

    if state.session_id:
        try:
            append_message(state.session_id, "assistant", response_message)
        except Exception as e:
            print(f"⚠️  Error storing escalation message in DynamoDB: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠️  session_id is None, skipping DynamoDB storage for escalation message")

    from langchain_core.messages import AIMessage

    messages = state.messages.copy() if state.messages else []
    if response_message:
        last_msg = messages[-1] if messages else None
        if not (
            last_msg
            and isinstance(last_msg, AIMessage)
            and last_msg.content == response_message
        ):
            messages.append(AIMessage(content=response_message))

    return {
        "response": response_message,
        "messages": messages,  
        "escalation_summary": escalation_message,
        "needs_contact_info": needs_contact_info,
        "escalation_id": escalation_id,
    }
