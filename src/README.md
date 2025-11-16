# Source Code Structure & Architecture

This document describes the internal structure, workflow, and configuration of the Customer Support Agent.

See the main project overview here: [README.md](../README.md).

## Project Structure

```text
src/
├── agent.py           # Main workflow orchestration using LangGraph
├── state.py           # Conversation state schema (Pydantic model)
├── nodes.py            # Individual processing nodes (classification, RAG, response generation, etc.)
├── prompts.py          # LLM prompt templates
├── config.py           # Environment configuration (LangSmith, API keys)
└── supabase_client.py  # Database client for escalation logging
```

## How It Works

### Workflow Architecture

The agent uses a **LangGraph StateGraph** with the following processing pipeline:

![Workflow Architecture](../static/image.png)

### Node Functions

#### 1. **classify** (`classify_intent`)

- **Purpose**: Categorizes user intent into one of: `billing`, `subscription`, `account`, `shipping`, `returns`, `general`

#### 2. **extract_info** (`extract_user_info`)

- **Purpose**: Extracts structured information from conversation (email, name, order_id, account_id)
- **Note**: Automatically updates Supabase escalation records if contact info is found later

#### 3. **rag** (`retrieve_context`)

- **Purpose**: Retrieves relevant policy documents using vector similarity search
- **Vector Store**: ChromaDB with OpenAI embeddings (`text-embedding-3-small`)
- **Retrieval**: Top 3 most similar chunks based on classification tag + user message
- **Output**: Sets `context` in state with retrieved policy text

#### 4. **response** (`generate_response`)

- **Purpose**: Generates customer service response based on policy context
- **Input**: Policy context, user message, conversation history (last 6 messages)
- **Logic**:
  - Asks for contact info if missing and issue type requires it (billing, subscription, account, returns)
  - Maintains conversation continuity
- **Output**: Sets `response` in state

#### 5. **response_validation** (`response_validation`)

- **Purpose**: Validates response quality and detects security emergencies
- **Validation Status**:
  - `PASS`: Response is acceptable
  - `RETRY`: Response needs regeneration (unfriendly/offensive)
  - `FAIL`: Security emergency detected or response would harm customer
- **Security Detection**: Automatically fails and escalates if user mentions:
  - Account hacked / security breach
  - Fraud / unauthorized charges
  - Identity theft / stolen account

#### 6. **update_messages** (`update_messages_node`)

- **Purpose**: Updates conversation history for checkpointing
- **Output**: Adds assistant response to `messages` list for persistence

#### 7. **escalate** (`escalation_node`)

- **Purpose**: Handles escalation to human support
- **Actions**:
  - Logs escalation data to Supabase
  - Generates escalation summary with full context
  - Prompts for email if contact info missing

## Current Settings

### Vector Store Configuration

- **Embedding Model**: `text-embedding-3-small` (OpenAI)
- **Vector Database**: Supabase (pgvector)
- **Storage Location**: Supabase `document_embeddings` table
- **Chunking**:
  - Chunk size: 1000 characters
  - Overlap: 100 characters
- **Retrieval**: Top 3 similar chunks (`k=3`)
- **Similarity Metric**: Cosine distance



### Retry Logic

- **Max Retries**: 3 attempts for response generation
- **Retry Trigger**: Validation returns `RETRY` status
- **After Max Retries**: Automatically escalates to human support

### Contact Information Collection

The agent requests contact information when:

- No email/name is available
- Issue type is: `billing`, `subscription`, `account`, or `returns`
- At least 2 messages have been exchanged (after initial exchange)

### Policy Document Structure

Policy documents are stored as PDFs in `document/` directory:

- Account Management Policy.pdf
- Billing & Payment.pdf
- Contact Information.pdf
- Customer Support Policy.pdf
- Shipping & Delivery Policy.pdf
- Subscription Management Policy.pdf

Documents are loaded into Supabase vector database (pgvector) for semantic search.
