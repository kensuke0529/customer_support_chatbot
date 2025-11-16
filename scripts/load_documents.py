#!/usr/bin/env python3
"""
Script to load PDF documents into Supabase vector database.
Run this after the migration has been applied.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nodes import doc_loader

def main():
    """Load all PDF documents into Supabase."""
    project_root = Path(__file__).parent.parent
    document_dir = project_root / "document"
    
    if not document_dir.exists():
        print(f"❌ Document directory not found: {document_dir}")
        print("   Please ensure the document directory exists with PDF files.")
        return
    
    # List of documents to load
    docs = [
        "Account Management Policy.pdf",
        "Billing & Payment.pdf",
        "Contact Information.pdf",
        "Customer Support Policy.pdf",
        "Shipping & Delivery Policy.pdf",
        "Subscription Management Policy.pdf",
    ]
    
    print("🚀 Starting document loading process...")
    print(f"📁 Document directory: {document_dir}\n")
    
    loaded_count = 0
    failed_count = 0
    
    for doc_name in docs:
        doc_path = document_dir / doc_name
        
        if not doc_path.exists():
            print(f"⚠️  Document not found: {doc_name}")
            failed_count += 1
            continue
        
        try:
            print(f"📄 Loading: {doc_name}...")
            chunk_count = doc_loader(str(doc_path), clear_existing=False)
            print(f"✅ Successfully loaded {doc_name} ({chunk_count} chunks)\n")
            loaded_count += 1
        except Exception as e:
            print(f"❌ Error loading {doc_name}: {e}\n")
            failed_count += 1
            import traceback
            traceback.print_exc()
    
    print("=" * 50)
    print(f"✅ Loaded: {loaded_count} documents")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count} documents")
    print("=" * 50)
    
    if loaded_count > 0:
        print("\n🎉 Documents loaded successfully!")
        print("   The application is now ready to use Supabase vector search.")
    else:
        print("\n⚠️  No documents were loaded. Please check the errors above.")

if __name__ == "__main__":
    main()

