"""
Purchase History DynamoDB Module
"""

import os
import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timezone, timedelta
from typing import List, Optional, TypedDict
from decimal import Decimal
import random

PURCHASE_TABLE_NAME = os.getenv("PURCHASE_HISTORY_TABLE", "customer-purchase-history")
REGION = os.getenv("AWS_REGION", "us-east-1")

# Lazy-initialize DynamoDB to avoid blocking startup
_dynamodb = None
_table = None


def get_table():
    """Lazy initialization of DynamoDB table."""
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb", region_name=REGION)
        _table = _dynamodb.Table(PURCHASE_TABLE_NAME)
    return _table


def table_exists() -> bool:
    """Check if the table exists."""
    try:
        dynamodb_resource = boto3.resource("dynamodb", region_name=REGION)
        table = dynamodb_resource.Table(PURCHASE_TABLE_NAME)
        table.load()
        return True
    except Exception:
        return False


def create_table_if_not_exists():
    """Create the table if it doesn't exist (for local testing)."""
    if table_exists():
        print(f"✅ Table {PURCHASE_TABLE_NAME} already exists")
        return

    try:
        print(f"📝 Creating table {PURCHASE_TABLE_NAME}...")
        dynamodb_resource = boto3.resource("dynamodb", region_name=REGION)
        table = dynamodb_resource.create_table(
            TableName=PURCHASE_TABLE_NAME,
            KeySchema=[
                {"AttributeName": "customer_email", "KeyType": "HASH"},
                {"AttributeName": "order_date", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "customer_email", "AttributeType": "S"},
                {"AttributeName": "order_date", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        # Wait for table to be created
        table.wait_until_exists()
        print(f"✅ Table {PURCHASE_TABLE_NAME} created successfully")
    except Exception as e:
        if "ResourceInUseException" in str(e):
            print(f"✅ Table {PURCHASE_TABLE_NAME} already exists")
        else:
            print(f"❌ Error creating table: {e}")
            raise


class PurchaseItem(TypedDict):
    product_name: str
    quantity: int
    unit_price: str
    total: str


class PurchaseRecord(TypedDict):
    customer_email: str
    order_id: str
    order_date: str
    status: str
    items: List[PurchaseItem]
    total_amount: str
    shipping_address: str
    tracking_number: Optional[str]


# Mock Data

MOCK_PRODUCTS = [
    {"name": "Premium Wireless Headphones", "price": 149.99},
    {"name": "Mechanical Keyboard RGB", "price": 89.99},
    {"name": "USB-C Hub 7-in-1", "price": 45.99},
    {"name": "Laptop Stand Adjustable", "price": 34.99},
    {"name": "Webcam HD 1080p", "price": 79.99},
    {"name": "Mouse Pad XL Gaming", "price": 24.99},
    {"name": "Monitor Light Bar", "price": 59.99},
    {"name": "Wireless Charger Fast", "price": 29.99},
    {"name": "Bluetooth Speaker Mini", "price": 39.99},
    {"name": "Cable Management Kit", "price": 19.99},
]

MOCK_CUSTOMERS = [
    {
        "email": "john.doe@email.com",
        "name": "John Doe",
        "address": "123 Main St, New York, NY 10001",
    },
    {
        "email": "jane.smith@email.com",
        "name": "Jane Smith",
        "address": "456 Oak Ave, Los Angeles, CA 90001",
    },
    {
        "email": "bob.wilson@email.com",
        "name": "Bob Wilson",
        "address": "789 Pine Rd, Chicago, IL 60601",
    },
    {
        "email": "alice.johnson@email.com",
        "name": "Alice Johnson",
        "address": "321 Elm St, Houston, TX 77001",
    },
    {
        "email": "test@example.com",
        "name": "Test User",
        "address": "100 Test Lane, San Francisco, CA 94102",
    },
]

ORDER_STATUSES = ["Processing", "Shipped", "Delivered", "Cancelled", "Refunded"]


def generate_order_id() -> str:
    """Generate a realistic order ID."""
    return f"ORD-{random.randint(100000, 999999)}"


def generate_tracking_number() -> str:
    """Generate a fake tracking number."""
    carriers = ["1Z", "94", "92"]
    prefix = random.choice(carriers)
    return f"{prefix}{random.randint(1000000000, 9999999999)}"


def create_mock_purchase(customer: dict, days_ago: int) -> PurchaseRecord:
    """Create a mock purchase record."""
    num_items = random.randint(1, 3)
    items = []
    total = Decimal("0")

    for _ in range(num_items):
        product = random.choice(MOCK_PRODUCTS)
        quantity = random.randint(1, 2)
        unit_price = Decimal(str(product["price"]))
        item_total = unit_price * quantity
        total += item_total

        items.append(
            {
                "product_name": product["name"],
                "quantity": quantity,
                "unit_price": str(unit_price),
                "total": str(item_total),
            }
        )

    order_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
    status = "Delivered" if days_ago > 5 else random.choice(["Processing", "Shipped"])

    return {
        "customer_email": customer["email"],
        "order_id": generate_order_id(),
        "order_date": order_date.isoformat(),
        "status": status,
        "items": items,
        "total_amount": str(total),
        "shipping_address": customer["address"],
        "tracking_number": generate_tracking_number()
        if status in ["Shipped", "Delivered"]
        else None,
    }


# Database Operations


def put_purchase(record: PurchaseRecord) -> None:
    """Store a purchase record in DynamoDB."""
    try:
        get_table().put_item(Item=record)
        print(f"✅ Stored purchase {record['order_id']} for {record['customer_email']}")
    except Exception as e:
        print(f"❌ Error storing purchase: {e}")
        raise


def get_purchases_by_email(email: str, limit: int = 10) -> List[PurchaseRecord]:
    """Get purchase history for a customer by email."""
    try:
        resp = get_table().query(
            KeyConditionExpression=Key("customer_email").eq(email.lower()),
            ScanIndexForward=False,  # newest first
            Limit=limit,
        )
        items = resp.get("Items", [])
        print(f"✅ Found {len(items)} purchases for {email}")
        return items
    except Exception as e:
        print(f"❌ Error querying purchases: {e}")
        return []


def get_purchase_by_order_id(order_id: str) -> Optional[PurchaseRecord]:
    """Get a specific order by order ID (requires scan since order_id is not partition key)."""
    try:
        from boto3.dynamodb.conditions import Attr
        
        # Normalize order ID
        order_id = order_id.upper().strip()
        if not order_id.startswith("ORD-"):
            order_id = f"ORD-{order_id}"
        
        # Use scan with filter - not ideal for large tables but works for demo
        # Using Attr condition instead of string expression for better compatibility
        resp = get_table().scan(
            FilterExpression=Attr("order_id").eq(order_id),
            Limit=10,  # Increase limit to ensure we find it
        )
        items = resp.get("Items", [])
        
        # If we got items but hit the limit, continue scanning
        while items and len([i for i in items if i.get("order_id") == order_id]) == 0:
            if "LastEvaluatedKey" not in resp:
                break
            resp = get_table().scan(
                FilterExpression=Attr("order_id").eq(order_id),
                ExclusiveStartKey=resp["LastEvaluatedKey"],
                Limit=10,
            )
            items.extend(resp.get("Items", []))
        
        # Find the exact match
        for item in items:
            if item.get("order_id") == order_id:
                print(f"✅ Found order {order_id}")
                return item
        
        print(f"⚠️ Order {order_id} not found")
        return None
    except Exception as e:
        print(f"❌ Error finding order: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_recent_purchases(email: str, days: int = 30) -> List[PurchaseRecord]:
    """Get purchases from the last N days for a customer."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    try:
        resp = get_table().query(
            KeyConditionExpression=Key("customer_email").eq(email.lower())
            & Key("order_date").gte(cutoff_str),
            ScanIndexForward=False,
        )
        items = resp.get("Items", [])
        print(f"✅ Found {len(items)} recent purchases for {email}")
        return items
    except Exception as e:
        print(f"❌ Error querying recent purchases: {e}")
        return []


# Mock Data Population


def populate_mock_data(num_orders_per_customer: int = 3) -> None:
    """Populate the database with mock purchase data."""
    print("🔄 Populating mock purchase data...")

    for customer in MOCK_CUSTOMERS:
        for i in range(num_orders_per_customer):
            days_ago = random.randint(1, 60)
            record = create_mock_purchase(customer, days_ago)
            put_purchase(record)

    print(f"✅ Created {len(MOCK_CUSTOMERS) * num_orders_per_customer} mock orders")


def ensure_mock_data_exists() -> None:
    """Check if mock data exists, create if not."""
    try:
        test_orders = get_purchases_by_email("test@example.com", limit=1)
        if not test_orders:
            print("No mock data found, populating...")
            populate_mock_data()
        else:
            print("Mock data already exists")
    except Exception as e:
        print(f"⚠️ Could not check/populate mock data: {e}")


# Formatted Output for Agent


def format_purchase_for_display(purchase: PurchaseRecord) -> str:
    """Format a purchase record for human-readable display."""
    items_str = "\n".join(
        f"  - {item['product_name']} x{item['quantity']} @ ${item['unit_price']} = ${item['total']}"
        for item in purchase.get("items", [])
    )

    tracking = purchase.get("tracking_number")
    tracking_str = f"Tracking: {tracking}" if tracking else "No tracking yet"

    return f"""
Order: {purchase["order_id"]}
Date: {purchase["order_date"][:10]}
Status: {purchase["status"]}
Items:
{items_str}
Total: ${purchase["total_amount"]}
Shipping: {purchase["shipping_address"]}
{tracking_str}
""".strip()


def format_purchase_summary(purchases: List[PurchaseRecord]) -> str:
    """Format multiple purchases as a summary."""
    if not purchases:
        return "No purchase history found."

    summaries = []
    for p in purchases[:5]:  # Limit to 5 for readability
        summaries.append(
            f"• {p['order_id']} ({p['order_date'][:10]}): {p['status']} - ${p['total_amount']}"
        )

    return "Recent Orders:\n" + "\n".join(summaries)
