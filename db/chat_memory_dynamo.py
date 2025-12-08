import os
import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timezone
from typing import List, Literal, TypedDict

DYNAMO_TABLE_NAME = os.getenv("CHAT_MESSAGES_TABLE", "chat_messages")
REGION = os.getenv("AWS_REGION", "us-east-1")

# Lazy-initialize DynamoDB to avoid blocking startup
_dynamodb = None
_table = None


def get_table():
    """Lazy initialization of DynamoDB table to avoid startup hangs."""
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb", region_name=REGION)
        _table = _dynamodb.Table(DYNAMO_TABLE_NAME)
    return _table


RoleType = Literal["user", "assistant", "human"]


class ChatMessage(TypedDict):
    session_id: str
    created_at: str
    role: RoleType
    content: str


def _now_iso() -> str:
    # sortable ISO timestamp
    return datetime.now(timezone.utc).isoformat()


def append_message(session_id: str, role: RoleType, content: str) -> None:
    if not session_id:
        print("⚠️  Cannot store message: session_id is None or empty")
        return

    item: ChatMessage = {
        "session_id": session_id,
        "created_at": _now_iso(),
        "role": role,
        "content": content,
    }
    try:
        get_table().put_item(Item=item)
        print(
            f"✅ Stored {role} message in DynamoDB (session_id: {session_id[:8]}..., table: {DYNAMO_TABLE_NAME})"
        )
    except Exception as e:
        print(f"❌ Error storing message in DynamoDB: {e}")
        print(f"   Table: {DYNAMO_TABLE_NAME}, Region: {REGION}")
        print(f"   Session ID: {session_id}")
        import traceback

        traceback.print_exc()
        raise  # Re-raise to see the actual error


def get_history(
    session_id: str,
    limit: int = 20,
) -> List[ChatMessage]:
    """Return last N messages in chronological order for a session."""
    resp = get_table().query(
        KeyConditionExpression=Key("session_id").eq(session_id),
        ScanIndexForward=False,  # newest first
        Limit=limit,
    )
    items = resp.get("Items", [])
    # reverse so oldest→newest
    return list(reversed(items))
