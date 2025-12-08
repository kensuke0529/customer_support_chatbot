#!/usr/bin/env python3
"""
AWS CDK Application Entry Point
===============================
Deploy the Customer Support AI Agent infrastructure using:
    cdk deploy --all
"""

import os
import aws_cdk as cdk
from stacks.agent_stack import AgentStack

app = cdk.App()

# Environment configuration
env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION", "us-east-1"),
)

# Deploy the main agent stack
AgentStack(
    app,
    "CustomerSupportAgentStack",
    env=env,
    description="Customer Support AI Agent - AWS App Runner deployment with DynamoDB",
)

app.synth()





