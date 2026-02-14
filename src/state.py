from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any
from langchain_core.messages import BaseMessage


class ChatbotInfo(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    user_message: str
    classification_tag: str
    context: str
    response: str
    response_validation: str
    response_validation_reason: str
    response_retry_count: int

    messages: List[BaseMessage] = []
    thread_id: Optional[str] = None

    user_email: Optional[str] = None
    user_name: Optional[str] = None
    order_id: Optional[str] = None
    session_id: Optional[str] = None
    contact_info_source: str = "none"
    needs_contact_info: bool = False
    escalation_id: Optional[str] = None
    has_asked_for_contact_info: bool = False
    contact_ask_count: int = 0

    email_extracted_this_turn: Optional[str] = None

    tool_calls_made: List[str] = []
    tool_results: Optional[str] = None
