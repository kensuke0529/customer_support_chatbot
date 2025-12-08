# db/escalations_dynamo.py
"""
DynamoDB-based escalation storage.
Replaces RDS for simpler serverless deployment.
"""

import os
import json
import uuid
import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timezone
from typing import Optional, Dict, Any

ESCALATIONS_TABLE = os.getenv("ESCALATIONS_TABLE", "escalations")
REGION = os.getenv("AWS_REGION", "us-east-1")

# Lazy-initialize DynamoDB to avoid blocking startup
_dynamodb = None
_table = None


def get_table():
    """Lazy initialization of DynamoDB table."""
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb", region_name=REGION)
        _table = _dynamodb.Table(ESCALATIONS_TABLE)
    return _table


def _now_iso() -> str:
    """Return sortable ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def create_escalation(
    user_message: str,
    classification_tag: Optional[str] = None,
    response: Optional[str] = None,
    user_email: Optional[str] = None,
    user_name: Optional[str] = None,
    order_id: Optional[str] = None,
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    contact_info_source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create an escalation record in DynamoDB.
    
    Returns:
        Dict with escalation_id and created_at
    """
    escalation_id = str(uuid.uuid4())
    created_at = _now_iso()
    
    item = {
        "escalation_id": escalation_id,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "pending",
        "user_message": user_message or "",
    }
    
    # Add optional fields if provided
    if classification_tag:
        item["classification_tag"] = classification_tag
        item["issue_type"] = classification_tag  # Alias for compatibility
    if response:
        item["response"] = response
    if user_email:
        item["user_email"] = user_email
    if user_name:
        item["user_name"] = user_name
    if order_id:
        item["order_id"] = order_id
    if session_id:
        item["session_id"] = session_id
    if thread_id:
        item["thread_id"] = thread_id
    if contact_info_source:
        item["contact_info_source"] = contact_info_source
    if metadata:
        item["metadata"] = json.dumps(metadata)
    
    try:
        get_table().put_item(Item=item)
        print(f"✅ Escalation created in DynamoDB (ID: {escalation_id[:8]}...)")
        return {"escalation_id": escalation_id, "created_at": created_at}
    except Exception as e:
        print(f"❌ Error creating escalation in DynamoDB: {e}")
        import traceback
        traceback.print_exc()
        raise


def update_escalation_with_contact_info(
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_email: Optional[str] = None,
    user_name: Optional[str] = None,
    order_id: Optional[str] = None,
    contact_info_source: Optional[str] = None,
) -> Optional[int]:
    """
    Update existing escalation records with contact information.
    Uses the session-index GSI to find escalations by session_id.
    
    Returns:
        Number of records updated, or None on error
    """
    if not session_id and not thread_id:
        print("⚠️  Cannot update escalation: no session_id or thread_id provided")
        return None
    
    try:
        table = get_table()
        
        # Query by session_id using GSI
        if session_id:
            response = table.query(
                IndexName="session-index",
                KeyConditionExpression=Key("session_id").eq(session_id),
                ScanIndexForward=False,  # Most recent first
                Limit=1
            )
        else:
            # If only thread_id, we need to scan (less efficient)
            response = table.scan(
                FilterExpression="thread_id = :tid",
                ExpressionAttributeValues={":tid": thread_id},
                Limit=1
            )
        
        items = response.get("Items", [])
        if not items:
            print("⚠️  No escalation found to update")
            return 0
        
        escalation = items[0]
        escalation_id = escalation["escalation_id"]
        created_at = escalation["created_at"]
        
        # Build update expression
        update_parts = ["#updated_at = :updated_at"]
        expr_names = {"#updated_at": "updated_at"}
        expr_values = {":updated_at": _now_iso()}
        
        if user_email:
            update_parts.append("#email = :email")
            expr_names["#email"] = "user_email"
            expr_values[":email"] = user_email
        if user_name:
            update_parts.append("#name = :name")
            expr_names["#name"] = "user_name"
            expr_values[":name"] = user_name
        if order_id:
            update_parts.append("#order = :order")
            expr_names["#order"] = "order_id"
            expr_values[":order"] = order_id
        if contact_info_source:
            update_parts.append("#source = :source")
            expr_names["#source"] = "contact_info_source"
            expr_values[":source"] = contact_info_source
        
        update_expr = "SET " + ", ".join(update_parts)
        
        table.update_item(
            Key={"escalation_id": escalation_id, "created_at": created_at},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values
        )
        
        print(f"✅ Updated escalation with contact info (ID: {escalation_id[:8]}...)")
        return 1
        
    except Exception as e:
        print(f"⚠️  Error updating escalation in DynamoDB: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_escalation_by_session(
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent escalation for a session/thread.
    
    Returns:
        Escalation record dict, or None if not found
    """
    if not session_id and not thread_id:
        return None
    
    try:
        table = get_table()
        
        if session_id:
            response = table.query(
                IndexName="session-index",
                KeyConditionExpression=Key("session_id").eq(session_id),
                ScanIndexForward=False,
                Limit=1
            )
        else:
            response = table.scan(
                FilterExpression="thread_id = :tid",
                ExpressionAttributeValues={":tid": thread_id},
                Limit=1
            )
        
        items = response.get("Items", [])
        return items[0] if items else None
        
    except Exception as e:
        print(f"⚠️  Error querying escalation from DynamoDB: {e}")
        return None




