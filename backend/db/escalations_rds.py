# db/escalations_rds.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
load_dotenv()

RDS_HOST = os.environ.get("RDS_HOST")
RDS_PORT = int(os.environ.get("RDS_PORT", 5432))
RDS_DB = os.environ.get("RDS_DB", "postgres")
RDS_USER = os.environ.get("RDS_USER")
RDS_PASSWORD = os.environ.get("RDS_PASSWORD")


def get_conn(dbname=None):
    """
    Get database connection with SSL support for RDS.

    Args:
        dbname: Optional database name. If not provided, uses RDS_DB env var or 'postgres'
    """
    if not RDS_HOST or not RDS_USER or not RDS_PASSWORD:
        raise ValueError(
            "Missing required RDS environment variables: RDS_HOST, RDS_USER, RDS_PASSWORD"
        )

    db_name = dbname or RDS_DB
    try:
        # RDS requires SSL connections - use sslmode='require' or 'prefer'
        return psycopg2.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            dbname=db_name,
            user=RDS_USER,
            password=RDS_PASSWORD,
            sslmode="require",  # RDS requires SSL encryption
            cursor_factory=RealDictCursor,
        )
    except psycopg2.OperationalError as e:
        error_str = str(e)
        if "does not exist" in error_str:
            raise ValueError(
                f"Database '{db_name}' does not exist on RDS instance. "
                f"Please create it or use an existing database name. "
                f"Error: {e}"
            )
        elif "password authentication failed" in error_str:
            raise ValueError(
                f"Password authentication failed for user '{RDS_USER}'. "
                f"Please check your RDS_USER and RDS_PASSWORD environment variables."
            )
        elif "no pg_hba.conf entry" in error_str and "no encryption" in error_str:
            raise ValueError(
                f"RDS requires SSL encryption. Connection attempted without SSL. "
                f"This should be fixed with sslmode='require', but if the error persists, "
                f"check your RDS security group settings. Error: {e}"
            )
        raise


def list_databases():
    """List all available databases on the RDS instance."""
    try:
        conn = get_conn(dbname="postgres")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname;"
            )
            return [row[0] for row in cur.fetchall()]
    except Exception:
        return []


def get_table_columns(table_name="escalations"):
    """Get column names and types for a table."""
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position;",
                    (table_name,),
                )
                return [
                    (col["column_name"], col["data_type"]) for col in cur.fetchall()
                ]
        finally:
            conn.close()
    except Exception as e:
        print(f"⚠️  Error getting table columns: {e}")
        import traceback

        traceback.print_exc()
        return []


def create_escalation(
    user_message: str,
    classification_tag: str | None,
    response: str | None,
    user_email: str | None = None,
    user_name: str | None = None,
    order_id: str | None = None,
    session_id: str | None = None,
    thread_id: str | None = None,
    contact_info_source: str | None = None,
    metadata: dict | None = None,
):
    """Insert into RDS escalations table. Automatically adapts to table schema."""
    columns = get_table_columns("escalations")
    if not columns:
        raise ValueError("Could not retrieve columns for 'escalations' table")

    column_names = {col[0] for col in columns}

    # Field mapping - maps function params to table columns
    field_mapping = {
        "user_message": user_message,
        "session_id": session_id,
        "issue_type": classification_tag,
        "classification_tag": classification_tag,
        "response": response,
        "status": "pending",
        "user_email": user_email,
        "user_name": user_name,
        "order_id": order_id,
        "thread_id": thread_id,
        "contact_info_source": contact_info_source,
        "metadata": json.dumps(metadata or {}) if metadata else None,
    }

    # Build insert columns and values
    insert_cols = []
    insert_values = []
    placeholders = []

    for col_name, value in field_mapping.items():
        if col_name in column_names and value is not None:
            insert_cols.append(col_name)
            insert_values.append(value)
            placeholders.append("%s")

    # Add timestamp columns
    for ts_col in ["created_at", "updated_at", "timestamp"]:
        if ts_col in column_names and ts_col not in insert_cols:
            insert_cols.append(ts_col)
            placeholders.append("NOW()")

    if not insert_cols:
        raise ValueError("No matching columns found in escalations table")

    # Build SQL - separate NOW() from parameterized values
    param_values = [v for v, p in zip(insert_values, placeholders) if p != "NOW()"]
    values_clause = ", ".join("NOW()" if p == "NOW()" else "%s" for p in placeholders)

    returning = (
        "escalation_id, created_at"
        if "escalation_id" in column_names and "created_at" in column_names
        else "*"
    )

    sql = f"INSERT INTO escalations ({', '.join(insert_cols)}) VALUES ({values_clause}) RETURNING {returning};"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(param_values))
            row = cur.fetchone()
            conn.commit()
    return row


def update_escalation_with_contact_info(
    session_id: str | None = None,
    thread_id: str | None = None,
    user_email: str | None = None,
    user_name: str | None = None,
    order_id: str | None = None,
    contact_info_source: str | None = None,
):
    """
    Updates an existing escalation record in RDS with contact information that was extracted later.

    Args:
        session_id: Session ID to find the escalation
        thread_id: Thread ID to find the escalation
        user_email: Email address to update
        user_name: Name to update
        order_id: Order ID to update
        contact_info_source: Source of the info (default: "extracted")

    Returns:
        Number of rows updated, or None if error
    """
    try:
        if not session_id and not thread_id:
            print("⚠️  Cannot update escalation: no session_id or thread_id provided")
            return None

        columns = get_table_columns("escalations")
        if not columns:
            print("⚠️  Could not retrieve columns for 'escalations' table")
            return None

        column_names = {col[0] for col in columns}

        # Build update data (only include non-None values and columns that exist)
        update_fields = {}
        if user_email and "user_email" in column_names:
            update_fields["user_email"] = user_email
        if user_name and "user_name" in column_names:
            update_fields["user_name"] = user_name
        if order_id and "order_id" in column_names:
            update_fields["order_id"] = order_id
        if contact_info_source and "contact_info_source" in column_names:
            update_fields["contact_info_source"] = contact_info_source

        if not update_fields:
            print("⚠️  No valid fields to update")
            return None

        # Add updated_at timestamp if column exists
        if "updated_at" in column_names:
            update_fields["updated_at"] = "NOW()"

        # Build UPDATE SQL
        set_clauses = []
        param_values = []

        for col_name, value in update_fields.items():
            if value == "NOW()":
                set_clauses.append(f"{col_name} = NOW()")
            else:
                set_clauses.append(f"{col_name} = %s")
                param_values.append(value)

        set_clause = ", ".join(set_clauses)

        # Build WHERE clause
        where_clauses = []
        if session_id and "session_id" in column_names:
            where_clauses.append("session_id = %s")
            param_values.append(session_id)
        if thread_id and "thread_id" in column_names:
            where_clauses.append("thread_id = %s")
            param_values.append(thread_id)

        if not where_clauses:
            print(
                "⚠️  Cannot build WHERE clause: session_id and thread_id columns not found"
            )
            return None

        where_clause = " AND ".join(where_clauses)

        # Try to update escalations that don't have email yet (null or empty)
        # First, try updating records where user_email is NULL or empty
        sql = f"""
            UPDATE escalations 
            SET {set_clause}
            WHERE {where_clause} 
            AND (user_email IS NULL OR user_email = '')
            RETURNING escalation_id, id;
        """

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(param_values))
                    row = cur.fetchone()
                    conn.commit()

                    if row:
                        escalation_id = row.get("escalation_id") or row.get("id")
                        print(
                            f"✅ Updated escalation with contact info (ID: {escalation_id})"
                        )
                        return 1
        except Exception as e:
            print(f"⚠️  Error in first update attempt: {e}")
            # Continue to fallback approach

        # If no rows updated (email already exists), find and update the most recent escalation
        # First, find the escalation ID
        try:
            find_sql = f"SELECT escalation_id, id FROM escalations WHERE {where_clause}"

            # Determine order by column
            order_by_col = None
            for col in ["created_at", "updated_at", "timestamp"]:
                if col in column_names:
                    order_by_col = col
                    break

            if order_by_col:
                find_sql += f" ORDER BY {order_by_col} DESC LIMIT 1"
            else:
                find_sql += " LIMIT 1"

            # Build params for finding (just WHERE clause params)
            find_params = []
            if session_id and "session_id" in column_names:
                find_params.append(session_id)
            if thread_id and "thread_id" in column_names:
                find_params.append(thread_id)

            with get_conn() as conn:
                with conn.cursor() as cur:
                    # Find the escalation
                    cur.execute(find_sql, tuple(find_params))
                    row = cur.fetchone()

                    if row:
                        escalation_id = row.get("escalation_id") or row.get("id")
                        id_col = (
                            "escalation_id" if "escalation_id" in column_names else "id"
                        )

                        # Now update it
                        update_sql = f"""
                            UPDATE escalations 
                            SET {set_clause}
                            WHERE {id_col} = %s
                            RETURNING escalation_id, id;
                        """

                        # Build params: update values + escalation_id
                        update_params = []
                        for col_name, value in update_fields.items():
                            if value != "NOW()":
                                update_params.append(value)
                        update_params.append(escalation_id)

                        cur.execute(update_sql, tuple(update_params))
                        updated_row = cur.fetchone()
                        conn.commit()

                        if updated_row:
                            print(
                                f"✅ Updated most recent escalation with contact info (ID: {escalation_id})"
                            )
                            return 1

            print("⚠️  No escalation found to update")
            return 0
        except Exception as e:
            print(f"⚠️  Error in fallback update attempt: {e}")
            import traceback

            traceback.print_exc()
            return 0  # Return 0 instead of None to indicate no update but no error

    except Exception as e:
        print(f"⚠️  Error updating escalation in RDS: {e}")
        import traceback

        traceback.print_exc()
        return None
