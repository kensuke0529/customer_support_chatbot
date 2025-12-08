# AWS Infrastructure - Customer Support AI Agent

This directory contains AWS CDK (Python) infrastructure code for deploying the Customer Support AI Agent to AWS App Runner.

## Architecture
![Workflow Architecture](/static/apprunner.png)

## Features
The Customer Support AI Agent AWS infrastructure provides:

- **No server management**: Runs on AWS App Runner (serverless), so there's nothing you need to patch or maintain.
- **Automatic scaling**: System grows or shrinks automatically based on how many customers are chatting.
- **Reliable data storage**: All chat history, escalations, and order data are securely stored in the cloud.

## Setup

### 1. Install CDK Dependencies

```bash
cd infra
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create API Keys Secret

Before deploying, create a Secrets Manager secret with your API keys:

```bash
aws secretsmanager create-secret \
    --name customer-support-agent/api-keys \
    --secret-string '{
        "OPENAI_API_KEY": "sk-your-openai-key",
        "LANGCHAIN_API_KEY": "lsv2_your-langsmith-key"
    }'
```

### 3. Bootstrap CDK (First Time Only)

```bash
cdk bootstrap aws://ACCOUNT-NUMBER/REGION
```

### 4. Deploy

```bash
# Preview changes
cdk diff

# Deploy
cdk deploy
```

