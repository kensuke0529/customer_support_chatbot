#!/usr/bin/env python3

import boto3
import uuid
import os
import numpy as np
from pathlib import Path
from nodes import doc_loader
from langchain_openai import OpenAIEmbeddings

# AWS 
REGION = "us-east-1"
VECTOR_BUCKET = "s3-vector-chatbot-policy-docs"
VECTOR_INDEX = "my-s3-vector-index"

s3v = boto3.client("s3vectors", region_name=REGION)

# Embedding model
openai_api_kex = os.getenv("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small", api_key=openai_api_key
)


def embed_text(text: str):
    embedding = embeddings_model.embed_query(text)
    vec = np.array(embedding, dtype=np.float32).tolist()
    return vec


def store_vector(text_chunk: str, metadata: dict):
    vec = embed_text(text_chunk)
    vector_obj = {
        "key": str(uuid.uuid4()),
        "data": {"float32": vec},
        "metadata": metadata,
    }
    s3v.put_vectors(
        vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX, vectors=[vector_obj]
    )


def load_all_docs():
    project_root = Path(__file__).parent.parent
    document_dir = project_root / "document"
    docs = [
        "Account Management Policy.pdf",
        "Billing & Payment.pdf",
        "Contact Information.pdf",
        "Customer Support Policy.pdf",
        "Shipping & Delivery Policy.pdf",
        "Subscription Management Policy.pdf",
    ]

    for doc_name in docs:
        doc_path = document_dir / doc_name
        chunks = doc_loader(str(doc_path), clear_existing=False, return_chunks=True)
        for i, chunk_text in enumerate(chunks):
            metadata = {
                "source_doc": doc_name,
                "chunk_index": i,
                "text": chunk_text,  
            }
            store_vector(chunk_text, metadata)
        print(f"Stored {len(chunks)} chunks from {doc_name}")


if __name__ == "__main__":
    load_all_docs()
    print("Done ingesting all documents to S3 Vectors.")
