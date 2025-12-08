# Customer Support Agent

An AI-powered customer support agent that automatically handles customer inquiries with accurate, policy-based responses and intelligent escalation.

**Deployed on AWS App Runner** - Serverless container hosting with automatic scaling



## Project Structure

```
chatbot-apprunner/
├── src/              # Core application logic
│   ├── agent.py      # LangGraph workflow orchestration
│   ├── nodes.py      # Processing nodes (classification, RAG, tools, etc.)
│   ├── state.py      # Conversation state schema
│   ├── prompts.py    # LLM prompt templates
│   ├── config.py     # Environment configuration
│   ├── tools.py      # Order lookup tools (DynamoDB)
│   └── load_documents.py  # Load PDFs into S3 Vectors
├── app/              # FastAPI application
│   └── main.py       # Production API entry point (App Runner)
├── db/               # Database clients
│   ├── chat_memory_dynamo.py      # DynamoDB chat history
│   ├── escalations_dynamo.py      # DynamoDB escalations
│   ├── purchase_history_dynamo.py # DynamoDB order data
│   └── escalations_rds.py         # RDS escalations (optional)
├── infra/            # AWS CDK infrastructure
│   ├── app.py        # CDK app entry point
│   ├── stacks/
│   │   └── agent_stack.py  # App Runner + DynamoDB stack
│   └── README.md     # Deployment guide
├── static/           # Frontend assets
├── document/         # Policy PDFs for RAG
├── scripts/          # Utility scripts
```

Read more technical details: [/src/README.md](./src/README.md).
## What It Does

The chatbot provides quick, accurate responses to customer questions about following categories:

- **Billing & Payment** - Charges, invoices, payment methods, pricing questions  
- **Subscription** - Starting, canceling, upgrading, or modifying subscriptions  
- **Account Issues** - Login problems, password resets, profile updates  
- **Shipping & Delivery** - Tracking information, delivery status, shipping updates  
- **Returns & Refunds** - Return processes, refund requests, damaged items  
- **General Inquiries** - Other customer service questions  

For more technical information (APIs, implementation details), see [/src/README.md](./src/README.md).

The chatbot extracts customer information (email, name, order IDs) from conversations, retrieves relevant policy information, generates responses that strictly adhere to company policies, and escalates complex or security-sensitive issues to human agents.

## Workflow Architecture
The agent follows this workflow for each customer inquiry:

### Key Components

1. **Intent Classification** - Categorizes inquiries (billing, subscription, account, shipping, returns, general)
2. **Customer Info Extraction** - Extracts email, name, order IDs from conversation (runs in parallel with classification)
3. **Tool Execution** - Queries DynamoDB purchase history table for order details (when relevant)
4. **RAG Retrieval** - Retrieves relevant policy documents from S3 Vectors using semantic search
5. **Response Generation** - Generates response using retrieved context, tool results, and conversation history
6. **Guardrails Validation** - LLM-based quality check (PASS/RETRY/FAIL) with automatic retry up to 3 times
7. **Emergency Detection** - Security keyword detection (hacked, fraud, unauthorized access) triggers immediate escalation
8. **Escalation Storage** - Stores escalation records in DynamoDB for human agent review

![Workflow Architecture](./static/image.png)
## Business Problem

Customer support teams face several challenges:

- **High Volume**: Large numbers of routine inquiries consume significant agent time
- **24/7 Demand**: Customers expect support availability outside business hours
- **Consistency**: Ensuring all agents provide accurate, policy-compliant responses
- **Resource Allocation**: Skilled agents spend time on repetitive questions instead of complex issues

## Solution

This AI agent addresses these challenges by:

1. **Instant Classification** - Automatically identifies the type of inquiry (billing, subscription, account, etc.)
2. **Policy-Based Responses** - Retrieves and uses exact company policy information from S3 Vectors (RAG) to ensure accuracy
3. **Order Lookup Tools** - Queries DynamoDB to retrieve customer purchase history and order details
4. **Quality Validation** - Validates each response for accuracy and professionalism before sending (with retry logic)
5. **Emergency Detection** - Automatically detects security emergencies (hacked accounts, fraud, unauthorized access)
6. **Intelligent Escalation** - Escalates security emergencies, validation failures, or complex issues to human agents
7. **Information Extraction** - Captures customer contact details and order information from natural conversation
8. **Conversation Memory** - Maintains context across multi-turn conversations using DynamoDB and LangGraph checkpointing

## Deployment

**Infrastructure**: AWS App Runner (serverless containers)
- **Container**: Docker image built from `dockerfile`
- **Database**: DynamoDB (chat memory, escalations, purchase history)
- **Vector Store**: S3 Vectors (policy documents for RAG)
- **Secrets**: AWS Secrets Manager (API keys)
- **Infrastructure as Code**: AWS CDK (Python)

![Workflow Architecture](./static/apprunner.png)

See [/infra/README.md](./infra/README.md) for deployment instructions.

### Architecture Notes

- **Database**: The system uses **DynamoDB** for all data storage (chat memory, escalations, purchase history). The flowchart may show "Postgres SQL" but the current implementation uses DynamoDB for serverless scalability. RDS/Postgres support exists as an optional fallback.
- **Tools**: Order lookup tools query DynamoDB purchase history table to provide real-time order information
- **RAG**: Policy documents are stored in S3 Vectors for semantic search and retrieval
- **Memory**: Conversation history is persisted in DynamoDB and managed via LangGraph checkpointing

## Business Results

Organizations using this agent experience:

- **Faster Response Times** - Customers receive instant answers instead of waiting in queues
- **Consistent Quality** - Every response adheres strictly to company policies, reducing errors
- **Reduced Costs** - Routine inquiries handled automatically, freeing agents for complex cases
- **24/7 Availability** - Customers get support anytime without additional staffing costs
