import re
from state import ChatbotInfo

# Direct loading of tools
try:
    from tools import get_tools
    tools = get_tools()
    print(f"Loaded {len(tools)} tools")
except Exception as e:
    print(f"Could not load tools: {e}")
    tools = []

def _should_use_tools(state: ChatbotInfo) -> bool:
    order_keywords = [
        "order", "purchase", "bought", "tracking", "shipment",
        "delivery", "where is my", "order status", "order history",
        "my orders", "recent orders", "ord-",
    ]
    message_lower = state.user_message.lower()
    return any(keyword in message_lower for keyword in order_keywords)

def _extract_order_id_from_message(message: str) -> str | None:
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
    if not _should_use_tools(state):
        print("Skipping tools - not an order-related query")
        return {"tool_results": None, "tool_calls_made": []}

    if not tools:
        print("No tools available")
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
            if ("status" in message_lower or "where" in message_lower or "track" in message_lower):
                if "check_order_status" in tool_map:
                    result = tool_map["check_order_status"].invoke(order_id)
                    tool_results.append(result)
                    tool_calls.append("check_order_status")
                    print(f"Called check_order_status for {order_id}")
            else:
                if "get_order_details" in tool_map:
                    result = tool_map["get_order_details"].invoke(order_id)
                    tool_results.append(result)
                    tool_calls.append("get_order_details")
                    print(f"Called get_order_details for {order_id}")
        elif state.user_email:
            if "lookup_customer_orders" in tool_map:
                result = tool_map["lookup_customer_orders"].invoke(state.user_email)
                tool_results.append(result)
                tool_calls.append("lookup_customer_orders")
                print(f"Called lookup_customer_orders for {state.user_email}")
        else:
            tool_results.append("To look up your order information, I'll need either your email address or order number (like ORD-123456).")
            tool_calls.append("info_needed")
            print("Need email or order ID to look up orders")
    except Exception as e:
        print(f"Tool execution error: {e}")
        import traceback
        traceback.print_exc()
        tool_results.append("Sorry, I encountered an error looking up that information.")

    combined_results = "\n\n".join(tool_results) if tool_results else None
    return {
        "tool_results": combined_results,
        "tool_calls_made": tool_calls,
    }
