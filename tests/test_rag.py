#!/usr/bin/env python3
"""
Test script to verify RAG (Retrieval-Augmented Generation) is working correctly.
This script checks:
1. If embeddings exist in Supabase
2. If retrieval function works
3. If relevant context is returned
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

load_dotenv()

from supabase_client import (
    check_embeddings_exist,
    search_similar_documents,
)
from nodes import retrieve_context
from langchain_openai import OpenAIEmbeddings
from state import ChatbotInfo

def test_embeddings_exist():
    """Test if embeddings exist in the database."""
    print("=" * 60)
    print("TEST 1: Checking if embeddings exist in Supabase")
    print("=" * 60)
    
    embeddings_exist = check_embeddings_exist()
    
    if embeddings_exist:
        print("✅ PASS: Embeddings found in Supabase")
        return True
    else:
        print("❌ FAIL: No embeddings found in Supabase")
        print("   💡 Run: python scripts/load_documents.py")
        return False

def test_embedding_generation():
    """Test if we can generate embeddings."""
    print("\n" + "=" * 60)
    print("TEST 2: Testing embedding generation")
    print("=" * 60)
    
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("❌ FAIL: OPENAI_API_KEY not found in environment")
            return False
        
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )
        
        test_query = "How do I change my email address?"
        query_embedding = embeddings_model.embed_query(test_query)
        
        if query_embedding and len(query_embedding) > 0:
            print(f"✅ PASS: Successfully generated embedding (dimension: {len(query_embedding)})")
            return True
        else:
            print("❌ FAIL: Generated embedding is empty")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Error generating embedding: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_search():
    """Test if vector search works."""
    print("\n" + "=" * 60)
    print("TEST 3: Testing vector similarity search")
    print("=" * 60)
    
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("❌ FAIL: OPENAI_API_KEY not found in environment")
            return False
        
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )
        
        # Test queries for different categories
        test_queries = [
            ("billing", "How do I update my payment method?"),
            ("account", "I want to change my email address"),
            ("shipping", "When will my order arrive?"),
            ("subscription", "How do I cancel my subscription?"),
        ]
        
        all_passed = True
        for classification_tag, user_message in test_queries:
            query_text = f"Question Category: {classification_tag} | User Question: {user_message}"
            query_embedding = embeddings_model.embed_query(query_text)
            
            # Try progressive thresholds like the actual node function
            thresholds = [0.5, 0.3, 0.2, 0.1]
            results = []
            
            for threshold in thresholds:
                results = search_similar_documents(
                    query_embedding=query_embedding,
                    match_threshold=threshold,
                    match_count=3,
                )
                if results:
                    break  # Found results, stop trying lower thresholds
            
            if results and len(results) > 0:
                print(f"✅ PASS: Found {len(results)} results for '{user_message}'")
                print(f"   Top result: {results[0].get('document_name', 'Unknown')} (similarity: {results[0].get('similarity', 0):.3f})")
                print(f"   Preview: {results[0].get('content', '')[:100]}...")
            else:
                print(f"❌ FAIL: No results found for '{user_message}'")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ FAIL: Error in vector search: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieve_context_node():
    """Test the full retrieve_context node function."""
    print("\n" + "=" * 60)
    print("TEST 4: Testing retrieve_context node function")
    print("=" * 60)
    
    try:
        # Test with different scenarios
        test_cases = [
            {
                "classification_tag": "billing",
                "user_message": "I need to update my credit card",
            },
            {
                "classification_tag": "account",
                "user_message": "How do I change my password?",
            },
        ]
        
        all_passed = True
        for test_case in test_cases:
            state = ChatbotInfo(
                user_message=test_case["user_message"],
                classification_tag=test_case["classification_tag"],
                context="",
                response="",
                response_validation="",
                response_validation_reason="",
                response_retry_count=0,
                contact_info_source="none",
                needs_contact_info=False,
                user_email=None,
                user_name=None,
                order_id=None,
                session_id=None,
                messages=[],
                thread_id="test-thread",
            )
            
            result = retrieve_context(state)
            context = result.get("context", "")
            
            if context and len(context) > 0:
                print(f"✅ PASS: Retrieved context for '{test_case['user_message']}'")
                print(f"   Context length: {len(context)} characters")
                print(f"   Preview: {context[:150]}...")
            else:
                print(f"❌ FAIL: No context retrieved for '{test_case['user_message']}'")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ FAIL: Error in retrieve_context: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all RAG tests."""
    print("\n" + "=" * 60)
    print("RAG SYSTEM TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Embeddings Exist", test_embeddings_exist()))
    results.append(("Embedding Generation", test_embedding_generation()))
    results.append(("Vector Search", test_vector_search()))
    results.append(("Retrieve Context Node", test_retrieve_context_node()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All RAG tests passed! RAG system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

