import os
import sys
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

_db_loaded = False
_get_purchases_by_email = None
_get_purchase_by_order_id = None
_format_purchase_for_display = None
_format_purchase_summary = None


def _load_purchase_db():
    """Lazy load purchase history database functions."""
    global _db_loaded, _get_purchases_by_email, _get_purchase_by_order_id
    global _format_purchase_for_display, _format_purchase_summary

    if _db_loaded:
        return

    try:
        from backend.db.purchase_history_dynamo import (
            get_purchases_by_email,
            get_purchase_by_order_id,
            format_purchase_for_display,
            format_purchase_summary,
        )

        _get_purchases_by_email = get_purchases_by_email
        _get_purchase_by_order_id = get_purchase_by_order_id
        _format_purchase_for_display = format_purchase_for_display
        _format_purchase_summary = format_purchase_summary
        _db_loaded = True
        print("Purchase history database loaded")
    except Exception as e:
        print(f"Could not load purchase history database: {e}")
        _db_loaded = True  


# Purchase History Tools


@tool
def lookup_customer_orders(customer_email: str) -> str:
    """
    Look up a customer's purchase history by their email address.
    Returns a summary of their recent orders including order IDs, dates, and status.

    Args:
        customer_email: The customer's email address (required)

    Returns:
        A formatted summary of the customer's recent orders, or an error message if not found.
    """
    _load_purchase_db()

    if not customer_email:
        return "I need your email address to look up your orders. Could you please provide it?"

    if _get_purchases_by_email is None:
        return "Sorry, I'm unable to access the order system at the moment. Please try again later."

    try:
        purchases = _get_purchases_by_email(customer_email.lower().strip(), limit=5)
        if not purchases:
            return f"I couldn't find any orders associated with the email '{customer_email}'. Please verify the email address or contact support if you believe this is an error."

        return _format_purchase_summary(purchases)
    except Exception as e:
        print(f"Error looking up orders: {e}")
        return "Sorry, I encountered an error looking up your orders. Please try again or contact support."


@tool
def get_order_details(order_id: str) -> str:
    """
    Get detailed information about a specific order by order ID.
    Returns complete order details including items, shipping, and tracking.

    Args:
        order_id
    Returns:
        Detailed order information including items, status, and tracking.
    """
    _load_purchase_db()

    if not order_id:
        return "I need an order ID to look up the order details. Order IDs look like 'ORD-123456'."

    if _get_purchase_by_order_id is None:
        return "Sorry, I'm unable to access the order system at the moment. Please try again later."

    order_id = order_id.upper().strip()
    if not order_id.startswith("ORD-"):
        order_id = f"ORD-{order_id}"

    try:
        order = _get_purchase_by_order_id(order_id)
        if not order:
            return f"I couldn't find an order with ID '{order_id}'. Please verify the order ID. It should be in the format 'ORD-123456'."

        return _format_purchase_for_display(order)
    except Exception as e:
        print(f"Error getting order details: {e}")
        return "Sorry, I encountered an error retrieving the order details. Please try again or contact support."


@tool
def check_order_status(order_id: str) -> str:
    """
    Quick check of order status and tracking information.

    Args:
        order_id: The order ID to check

    Returns:
        Current status and tracking information for the order.
    """
    _load_purchase_db()

    if not order_id:
        return "I need an order ID to check the status. Do you have your order number?"

    if _get_purchase_by_order_id is None:
        return "Sorry, I'm unable to access the order system at the moment. Please try again later."

    order_id = order_id.upper().strip()
    if not order_id.startswith("ORD-"):
        order_id = f"ORD-{order_id}"

    try:
        order = _get_purchase_by_order_id(order_id)
        if not order:
            return f"I couldn't find order '{order_id}'. Please check the order ID and try again."

        status = order.get("status", "Unknown")
        tracking = order.get("tracking_number")
        order_date = order.get("order_date", "")[:10]

        response = f"Order {order_id} (placed {order_date}):\n"
        response += f"Status: {status}\n"

        if status == "Delivered":
            response += "Your order has been delivered!"
        elif status == "Shipped" and tracking:
            response += f"Tracking number: {tracking}\n"
            response += "Your package is on its way!"
        elif status == "Processing":
            response += "Your order is being prepared for shipment."
        elif status == "Cancelled":
            response += "This order was cancelled."
        elif status == "Refunded":
            response += "This order has been refunded."

        return response
    except Exception as e:
        print(f"Error checking order status: {e}")
        return "Sorry, I couldn't check the order status. Please try again."


# All available tools for the agent
CUSTOMER_SUPPORT_TOOLS = [
    lookup_customer_orders,
    get_order_details,
    check_order_status,
]


def get_tools():
    """Return the list of available tools for the agent."""
    return CUSTOMER_SUPPORT_TOOLS




