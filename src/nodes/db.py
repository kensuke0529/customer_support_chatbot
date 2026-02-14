from .base import openai_api_key

try:
    from backend.db.chat_memory_dynamo import append_message as _am, get_history as _gh
    append_message = _am
    get_history = _gh
    print("DynamoDB chat history enabled")
except Exception as e:
    print(f"DynamoDB chat history import failed: {e}")
    append_message = lambda *args, **kwargs: None
    get_history = lambda *args, **kwargs: []

try:
    from backend.db.escalations_dynamo import (
        create_escalation as _ce,
        update_escalation_with_contact_info as _ueci,
        get_escalation_by_session as _gebs,
    )
    create_escalation = _ce
    update_escalation_with_contact_info = _ueci
    get_escalation_by_session = _gebs
    USE_DYNAMO_ESCALATIONS = True
    print("DynamoDB escalations enabled")
    
except Exception as e:
    print(f"DynamoDB escalations import failed: {e}")
    print("Escalations will be logged to console only")
    USE_DYNAMO_ESCALATIONS = False
    create_escalation = lambda **kwargs: {"escalation_id": "console-only"}
    update_escalation_with_contact_info = lambda **kwargs: None
    get_escalation_by_session = lambda **kwargs: None

def is_dynamo_escalations_enabled():
    return USE_DYNAMO_ESCALATIONS
