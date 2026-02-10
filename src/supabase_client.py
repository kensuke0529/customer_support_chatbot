from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_API_KEY")


if url and key:
    try:
        supabase = create_client(url, key)
        print("Supabase client initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}")
else:
    print(
        "Warning: SUPABASE_URL or SUPABASE_API_KEY not found in environment variables"
    )


def add_data_to_supabase(data: dict):
    """
    Adds escalation data to Supabase.
    
    Args:
        data: Dictionary containing escalation data including:
            - timestamp, user_message, classification_tag, response
            - validation_reason, retry_count, context_preview
            - user_email, user_name, order_id (extracted info)
            - session_id, thread_id, contact_info_source
    """
    try:
        if supabase:
            supabase.table("escalations").insert(data).execute()
            print(f"✅ Escalation data saved to Supabase")
        else:
            print("⚠️  Supabase client not initialized, skipping database save")
    except Exception as e:
        error_msg = str(e)
        if "column" in error_msg.lower() and "schema" in error_msg.lower():
            print(f"⚠️  Supabase schema error: Missing columns in 'escalations' table")
            print(f"   Error: {error_msg}")
            print(f"   💡 Run the migration SQL from 'supabase_migration.sql'")
            print(f"   💡 Or see 'README_SUPABASE.md' for instructions")
        else:
            print(f"⚠️  Error saving to Supabase: {error_msg}")


def update_escalation_with_contact_info(session_id: str, thread_id: str, user_email: str = None, user_name: str = None, order_id: str = None, contact_info_source: str = "extracted"):
    """
    Updates an existing escalation record with contact information that was extracted later.
    
    Args:
        session_id: Session ID to find the escalation
        thread_id: Thread ID to find the escalation
        user_email: Email address to update
        user_name: Name to update
        order_id: Order ID to update
        contact_info_source: Source of the info (default: "extracted")
    """
    try:
        if not supabase:
            return
        
        # Build update data (only include non-None values)
        update_data = {}
        if user_email:
            update_data["user_email"] = user_email
        if user_name:
            update_data["user_name"] = user_name
        if order_id:
            update_data["order_id"] = order_id
        if contact_info_source:
            update_data["contact_info_source"] = contact_info_source
        
        if not update_data:
            return  # Nothing to update
        
        # Find and update escalation by session_id or thread_id
        # Update escalations that don't have email yet (null or empty)
        if session_id:
            # Try to find escalation with this session_id that has no email
            result = supabase.table("escalations").update(update_data).eq("session_id", session_id).is_("user_email", "null").execute()
            if result.data and len(result.data) > 0:
                print(f"✅ Updated escalation with contact info (session_id: {session_id})")
                return
            
            # If that didn't work, try updating the most recent escalation for this session
            result = supabase.table("escalations").select("id").eq("session_id", session_id).order("timestamp", desc=True).limit(1).execute()
            if result.data and len(result.data) > 0:
                escalation_id = result.data[0]["id"]
                supabase.table("escalations").update(update_data).eq("id", escalation_id).execute()
                print(f"✅ Updated escalation with contact info (session_id: {session_id})")
                return
        
        if thread_id:
            # Try to find escalation with this thread_id that has no email
            result = supabase.table("escalations").update(update_data).eq("thread_id", thread_id).is_("user_email", "null").execute()
            if result.data and len(result.data) > 0:
                print(f"✅ Updated escalation with contact info (thread_id: {thread_id})")
                return
            
            # If that didn't work, try updating the most recent escalation for this thread
            result = supabase.table("escalations").select("id").eq("thread_id", thread_id).order("timestamp", desc=True).limit(1).execute()
            if result.data and len(result.data) > 0:
                escalation_id = result.data[0]["id"]
                supabase.table("escalations").update(update_data).eq("id", escalation_id).execute()
                print(f"✅ Updated escalation with contact info (thread_id: {thread_id})")
                return
        
    except Exception as e:
        error_msg = str(e)
        # Don't print error if it's just "no rows to update"
        if "null" not in error_msg.lower() and "no rows" not in error_msg.lower():
            print(f"⚠️  Error updating escalation: {error_msg}")


# Vector Database Operations


def add_embeddings_to_supabase(embeddings_data: list):
    """
    Adds document embeddings to Supabase vector database.
    
    Args:
        embeddings_data: List of dictionaries containing:
            - content: Text content of the chunk
            - embedding: Vector embedding (list of floats)
            - document_name: Name of the source document
            - chunk_index: Index of the chunk within the document
            - metadata: Optional metadata dictionary
    """
    try:
        if not supabase:
            print("⚠️  Supabase client not initialized, skipping embedding save")
            return False
        
        # Insert embeddings in batches for better performance
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i:i + batch_size]
            
            # Prepare data for insertion
            insert_data = []
            for item in batch:
                row = {
                    "content": item["content"],
                    "embedding": item["embedding"],  # Should be a list of floats
                    "document_name": item.get("document_name", ""),
                    "chunk_index": item.get("chunk_index", 0),
                    "metadata": item.get("metadata", {})
                }
                insert_data.append(row)
            
            # Insert batch
            result = supabase.table("document_embeddings").insert(insert_data).execute()
            total_inserted += len(insert_data)
            print(f"✅ Inserted {len(insert_data)} embeddings (total: {total_inserted}/{len(embeddings_data)})")
        
        print(f"✅ All embeddings saved to Supabase ({total_inserted} total)")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  Error saving embeddings to Supabase: {error_msg}")
        return False


def search_similar_documents(query_embedding: list, match_threshold: float = 0.7, match_count: int = 3, filter_document_name: str = None):
    """
    Searches for similar documents using vector similarity.
    
    Args:
        query_embedding: Vector embedding of the query (list of floats)
        match_threshold: Minimum similarity threshold (0-1)
        match_count: Maximum number of results to return
        filter_document_name: Optional document name to filter by
    
    Returns:
        List of dictionaries containing:
            - content: Text content
            - document_name: Name of the document
            - similarity: Similarity score
            - metadata: Additional metadata
    """
    try:
        if not supabase:
            print("⚠️  Supabase client not initialized, returning empty results")
            return []
        
        # Call the match_documents function via RPC
        params = {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count
        }
        
        if filter_document_name:
            params["filter_document_name"] = filter_document_name
        
        result = supabase.rpc("match_documents", params).execute()
        
        if result.data:
            return result.data
        return []
        
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  Error searching documents: {error_msg}")
        return []


def clear_all_embeddings():
    """
    Clears all document embeddings from the database.
    Useful for re-indexing documents.
    """
    try:
        if not supabase:
            print("⚠️  Supabase client not initialized")
            return False
        
        # Call the clear_all_embeddings function via RPC
        result = supabase.rpc("clear_all_embeddings").execute()
        print("✅ All embeddings cleared from Supabase")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  Error clearing embeddings: {error_msg}")
        return False


def check_embeddings_exist(document_name: str = None):
    """
    Checks if embeddings exist in the database.
    
    Args:
        document_name: Optional document name to check for
    
    Returns:
        Boolean indicating if embeddings exist
    """
    try:
        if not supabase:
            return False
        
        query = supabase.table("document_embeddings").select("id", count="exact")
        
        if document_name:
            query = query.eq("document_name", document_name)
        
        result = query.limit(1).execute()
        
        # Check if count is available or if we got any results
        if hasattr(result, 'count') and result.count:
            return result.count > 0
        elif result.data:
            return len(result.data) > 0
        
        return False
        
    except Exception as e:
        print(f"⚠️  Error checking embeddings: {e}")
        return False


def clear_document_embeddings(document_name: str):
    """
    Clears embeddings for a specific document.
    
    Args:
        document_name: Name of the document to clear
    
    Returns:
        Boolean indicating success
    """
    try:
        if not supabase:
            print("⚠️  Supabase client not initialized")
            return False
        
        # Call the clear_document_embeddings function via RPC
        result = supabase.rpc("clear_document_embeddings", {"document_name_to_clear": document_name}).execute()
        print(f"✅ Cleared embeddings for {document_name}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  Error clearing document embeddings: {error_msg}")
        return False
