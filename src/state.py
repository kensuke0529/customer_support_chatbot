from pydantic import BaseModel, ConfigDict
from typing import List, Optional
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

    # Memory/Conversation for checkpointing
    messages: List[BaseMessage] = []  # Full conversation history
    thread_id: Optional[str] = None  # For checkpointing conversation state

    # User information extraction
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    order_id: Optional[str] = None
    session_id: Optional[str] = None  # For conversation tracking
    contact_info_source: str = "none"  # "extracted", "provided", "none"
    needs_contact_info: bool = False  # Flag to prompt for email on escalation
