#!/usr/bin/env python3
"""
Script to populate the purchase history DynamoDB table with mock data.
Run this after deploying the CDK stack to populate test data.

Usage:
    python scripts/populate_mock_purchases.py

Environment Variables:
    AWS_REGION: AWS region (default: us-east-1)
    PURCHASE_HISTORY_TABLE: Table name (default: customer-purchase-history)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.purchase_history_dynamo import (
    populate_mock_data,
    ensure_mock_data_exists,
    get_purchases_by_email,
    create_table_if_not_exists,
    MOCK_CUSTOMERS,
)


def main():
    print("=" * 60)
    print("Purchase History Mock Data Population Script")
    print("=" * 60)
    print(f"Region: {os.getenv('AWS_REGION', 'us-east-1')}")
    print(f"Table: {os.getenv('PURCHASE_HISTORY_TABLE', 'customer-purchase-history')}")
    print()

    # Create table if it doesn't exist
    print("Checking if table exists...")
    try:
        create_table_if_not_exists()
    except Exception as e:
        print(f"⚠️  Warning: Could not create table (may already exist): {e}")
        print("Continuing anyway...")

    # Check if data already exists
    print("\nChecking for existing data...")
    try:
        ensure_mock_data_exists()
    except Exception as e:
        print(f"Error checking/creating mock data: {e}")
        print("\nNote: Make sure the DynamoDB table exists and you have AWS credentials configured.")
        print("If testing locally, ensure AWS credentials are set or use DynamoDB Local.")
        sys.exit(1)

    # Display sample data
    print("\n" + "=" * 60)
    print("Sample Data Created:")
    print("=" * 60)

    for customer in MOCK_CUSTOMERS[:3]:  # Show first 3 customers
        email = customer["email"]
        print(f"\n📧 Customer: {email}")
        orders = get_purchases_by_email(email, limit=2)
        for order in orders:
            print(f"   └─ {order['order_id']} | {order['order_date'][:10]} | {order['status']} | ${order['total_amount']}")

    print("\n" + "=" * 60)
    print("✅ Mock data ready! You can now test order lookups.")
    print("=" * 60)
    print("\nTest emails you can use:")
    for customer in MOCK_CUSTOMERS:
        print(f"  - {customer['email']}")


if __name__ == "__main__":
    main()

