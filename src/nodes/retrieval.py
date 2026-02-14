import os
import boto3
import numpy as np
from langchain_openai import OpenAIEmbeddings
from state import ChatbotInfo
from .base import openai_api_key

S3_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_VECTOR_BUCKET = os.getenv("VECTOR_BUCKET", "s3-vector-chatbot-policy-docs")
S3_VECTOR_INDEX = os.getenv("VECTOR_INDEX", "my-s3-vector-index")

# Direct initialization of S3 Vectors client
s3v_client = boto3.client("s3vectors", region_name=S3_REGION)

def retrieve_context(state: ChatbotInfo):
    try:
        return retrieve_context_rag(state)
    except Exception as e:
        error_msg = f"RAG FAILED: S3 Vectors retrieval failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"context": ""}

def retrieve_context_rag(state: ChatbotInfo, top_k: int = 5):
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=openai_api_key
    )

    query_text = f"{state.classification_tag} question: {state.user_message}"
    query_embedding = embeddings_model.embed_query(query_text)
    query_vec = np.array(query_embedding, dtype=np.float32).tolist()

    try:
        resp = s3v_client.query_vectors(
            vectorBucketName=S3_VECTOR_BUCKET,
            indexName=S3_VECTOR_INDEX,
            queryVector={"float32": query_vec},
            topK=top_k,
            returnMetadata=True,
        )

        hits = resp.get("vectors", [])
        if not hits:
            print("No similar vectors found via S3 Vectors")
            return {"context": ""}

        context_chunks = []
        for hit in hits:
            meta = hit.get("metadata", {})
            text = meta.get("text")
            if text:
                context_chunks.append(text)
            else:
                doc_name = meta.get("source_doc")
                chunk_idx = meta.get("chunk_index")
                if doc_name and chunk_idx is not None:
                    print(f"Text not found in metadata for {doc_name} chunk {chunk_idx}")

        if context_chunks:
            context = "\n\n".join(context_chunks)
            print(f"Retrieved {len(context_chunks)} chunks from S3 Vectors")
            return {"context": context}
        else:
            print("No text content found in retrieved vectors")
            return {"context": ""}

    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"RAG ERROR: Failed to query S3 Vectors ({error_type}): {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"context": ""}
