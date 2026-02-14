import json
from langchain_openai import ChatOpenAI
from state import ChatbotInfo
from .base import openai_api_key
from .escalation import log_escalation

# Direct initialization of validation LLM
validation_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
)

def _fast_security_check(user_message: str) -> bool:
    security_keywords = [
        "hacked", "hack", "fraud", "fraudulent", "unauthorized",
        "security breach", "stolen", "identity theft", "someone else",
        "not me", "didn't authorize", "suspicious activity", "compromised",
    ]
    message_lower = user_message.lower()
    return any(keyword in message_lower for keyword in security_keywords)

def _fast_response_check(response: str) -> tuple[bool, str]:
    if not response or len(response.strip()) < 10:
        return False, "Response is empty or too short"

    bad_patterns = [
        "i don't know", "i cannot help", "error occurred",
        "something went wrong", "undefined", "null",
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
        print(f"Fast validation failed: {reason}")
        return {
            "response_validation": "RETRY",
            "response_validation_reason": reason,
            "response_retry_count": state.response_retry_count + 1,
        }

    simple_categories = ["general"]
    if state.classification_tag in simple_categories and len(state.response) > 30:
        print("Fast validation PASS for simple query")
        return {
            "response_validation": "PASS",
            "response_validation_reason": "Fast path: simple query with adequate response",
            "response_retry_count": state.response_retry_count,
        }

    validation_prompt = f"""You are validating a customer support response. Be lenient and practical.

User Question: {state.user_message}
Generated Response: {state.response}
Context Available: {state.context[:300]}...

Respond in JSON format:
{{
    "status": "PASS" | "RETRY" | "FAIL",
    "reason": "Brief explanation"
}}


FAIL if the user question mentions ANY of these:
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
- No security emergency mentioned
- Response provides helpful information or guidance, even if partial

RETRY if:
- Response is not friendly or helpful 
- Response is completely unhelpful or nonsensical

FAIL if:
- Security emergency detected (see above)
- Response would harm the customer

IMPORTANT: If a response mentions "escalate" or "I'll escalate", this is still a valid response attempt and should PASS validation. The system will handle actual escalation separately. Only fail if the response is truly unhelpful, offensive, or harmful.
"""

    try:
        response = validation_llm.invoke(validation_prompt)
        content = response.content.strip()

        if content.startswith("```"):
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
            security_keywords = ["hacked", "fraud", "unauthorized", "security", "breach", "stolen"]
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

    except Exception as e:
        print(f"Validation error: {e}")
        return {
            "response_validation": "FAIL",
            "response_validation_reason": f"Validation process error: {str(e)}",
            "response_retry_count": state.response_retry_count,
        }
