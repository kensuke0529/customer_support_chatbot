from langchain_core.messages import AIMessage
from state import ChatbotInfo

def update_messages_node(state: ChatbotInfo):
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
