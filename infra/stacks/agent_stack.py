"""
Customer Support AI Agent - AWS CDK Stack
==========================================
Infrastructure as Code for deploying the LangGraph agent on AWS.

Components:
- AWS App Runner: Serverless container hosting
- DynamoDB: Chat memory persistence
- IAM: Service roles and permissions
- Secrets Manager: API key storage
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_apprunner as apprunner,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    aws_dynamodb as dynamodb,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class AgentStack(Stack):
    """
    Main infrastructure stack for the Customer Support AI Agent.

    This stack creates:
    1. ECR repository and Docker image build
    2. App Runner service with health checks
    3. DynamoDB table for chat memory
    4. IAM roles with least-privilege permissions
    5. Secrets Manager for API keys
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # =====================================================================
        # DynamoDB Table for Chat Memory (use existing table)
        # =====================================================================
        # The table was created previously with RETAIN policy, so we import it
        chat_memory_table = dynamodb.Table.from_table_name(
            self,
            "ChatMemoryTable",
            table_name="customer-support-chat-memory",
        )

        # =====================================================================
        # DynamoDB Table for Purchase History
        # =====================================================================
        # Import existing table (already created with mock data)
        # This avoids conflicts and preserves existing data
        purchase_history_table = dynamodb.Table.from_table_name(
            self,
            "PurchaseHistoryTable",
            table_name="customer-purchase-history",
        )

        # =====================================================================
        # Secrets Manager for API Keys
        # =====================================================================
        # Note: Create this secret manually in AWS Console or via CLI with your actual keys
        # aws secretsmanager create-secret --name customer-support-agent/api-keys \
        #     --secret-string '{"OPENAI_API_KEY":"sk-...","LANGCHAIN_API_KEY":"lsv2_..."}'
        # Using ARN directly to avoid CloudFormation validation hook issues
        secret_arn = f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:customer-support-agent/api-keys-*"
        api_keys_secret = secretsmanager.Secret.from_secret_complete_arn(
            self,
            "ApiKeysSecret",
            secret_complete_arn="arn:aws:secretsmanager:us-east-1:291480921130:secret:customer-support-agent/api-keys-v6FoQG",
        )

        # =====================================================================
        # IAM Role for App Runner
        # =====================================================================
        instance_role = iam.Role(
            self,
            "AppRunnerInstanceRole",
            assumed_by=iam.ServicePrincipal("tasks.apprunner.amazonaws.com"),
            description="IAM role for Customer Support Agent App Runner instance",
        )

        # Grant DynamoDB access
        chat_memory_table.grant_read_write_data(instance_role)
        purchase_history_table.grant_read_write_data(instance_role)

        # Grant Secrets Manager access
        api_keys_secret.grant_read(instance_role)

        # Grant S3 Vectors permissions for RAG
        # Using wildcard for S3 Vectors as the ARN format may vary
        instance_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3vectors:*",
                ],
                resources=["*"],
            )
        )

        # =====================================================================
        # IAM Role for ECR Access
        # =====================================================================
        access_role = iam.Role(
            self,
            "AppRunnerAccessRole",
            assumed_by=iam.ServicePrincipal("build.apprunner.amazonaws.com"),
            description="IAM role for App Runner to access ECR",
        )

        access_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSAppRunnerServicePolicyForECRAccess"
            )
        )

        # =====================================================================
        # Docker Image Build (ECR Asset)
        # =====================================================================
        # Build and push Docker image to ECR
        # IMPORTANT: platform=LINUX_AMD64 fixes Apple Silicon compatibility
        image_asset = ecr_assets.DockerImageAsset(
            self,
            "AgentImage",
            directory="..",  # Point to project root (parent of infra/)
            platform=ecr_assets.Platform.LINUX_AMD64,  # Critical for M1/M2 Macs
            exclude=[
                "infra",
                "cdk.out",
                ".git",
                "__pycache__",
                "*.pyc",
                ".env",
                ".venv",
                "venv",
            ],
        )

        # =====================================================================
        # App Runner Service
        # =====================================================================
        service = apprunner.CfnService(
            self,
            "AgentService",
            service_name="customer-support-agent",
            source_configuration=apprunner.CfnService.SourceConfigurationProperty(
                authentication_configuration=apprunner.CfnService.AuthenticationConfigurationProperty(
                    access_role_arn=access_role.role_arn,
                ),
                auto_deployments_enabled=False,  # Manual deployments for control
                image_repository=apprunner.CfnService.ImageRepositoryProperty(
                    image_identifier=image_asset.image_uri,
                    image_repository_type="ECR",
                    image_configuration=apprunner.CfnService.ImageConfigurationProperty(
                        port="8080",
                        runtime_environment_variables=[
                            # AWS Region for SDK
                            apprunner.CfnService.KeyValuePairProperty(
                                name="AWS_REGION",
                                value=self.region,
                            ),
                            # DynamoDB table names
                            apprunner.CfnService.KeyValuePairProperty(
                                name="DYNAMODB_TABLE_NAME",
                                value=chat_memory_table.table_name,
                            ),
                            apprunner.CfnService.KeyValuePairProperty(
                                name="PURCHASE_HISTORY_TABLE",
                                value=purchase_history_table.table_name,
                            ),
                            # LangSmith Tracing (observability)
                            apprunner.CfnService.KeyValuePairProperty(
                                name="LANGCHAIN_TRACING_V2",
                                value="true",
                            ),
                            apprunner.CfnService.KeyValuePairProperty(
                                name="LANGCHAIN_PROJECT",
                                value="customer-support-agent",
                            ),
                            # S3 Vectors configuration for RAG
                            apprunner.CfnService.KeyValuePairProperty(
                                name="VECTOR_BUCKET",
                                value="s3-vector-chatbot-policy-docs",
                            ),
                            apprunner.CfnService.KeyValuePairProperty(
                                name="VECTOR_INDEX",
                                value="my-s3-vector-index",
                            ),
                        ],
                        runtime_environment_secrets=[
                            # API keys from Secrets Manager
                            apprunner.CfnService.KeyValuePairProperty(
                                name="OPENAI_API_KEY",
                                value=f"{api_keys_secret.secret_arn}:OPENAI_API_KEY::",
                            ),
                            apprunner.CfnService.KeyValuePairProperty(
                                name="LANGCHAIN_API_KEY",
                                value=f"{api_keys_secret.secret_arn}:LANGCHAIN_API_KEY::",
                            ),
                        ],
                    ),
                ),
            ),
            instance_configuration=apprunner.CfnService.InstanceConfigurationProperty(
                cpu="1024",  # 1 vCPU
                memory="2048",  # 2 GB RAM
                instance_role_arn=instance_role.role_arn,
            ),
            health_check_configuration=apprunner.CfnService.HealthCheckConfigurationProperty(
                protocol="TCP",  # TCP is more forgiving during container startup
                interval=10,  # Check every 10 seconds
                timeout=5,  # 5 second timeout
                healthy_threshold=1,  # 1 success = healthy
                unhealthy_threshold=10,  # 10 failures = ~100 seconds startup grace period
            ),
            # auto_scaling_configuration_arn=self._create_auto_scaling_config(),  # Temporarily disabled to avoid validation issues
        )

        # =====================================================================
        # Outputs
        # =====================================================================
        CfnOutput(
            self,
            "ServiceUrl",
            value=f"https://{service.attr_service_url}",
            description="App Runner Service URL",
        )

        CfnOutput(
            self,
            "HealthCheckUrl",
            value=f"https://{service.attr_service_url}/health",
            description="Health Check Endpoint",
        )

        CfnOutput(
            self,
            "ApiDocsUrl",
            value=f"https://{service.attr_service_url}/docs",
            description="FastAPI Swagger Documentation",
        )

        CfnOutput(
            self,
            "DynamoDBTableName",
            value=chat_memory_table.table_name,
            description="DynamoDB Table for Chat Memory",
        )

    def _create_auto_scaling_config(self) -> str:
        """Create auto-scaling configuration for cost optimization."""
        auto_scaling_config = apprunner.CfnAutoScalingConfiguration(
            self,
            "AutoScalingConfig",
            auto_scaling_configuration_name="customer-support-agent-scaling",
            max_concurrency=100,  # Max concurrent requests per instance
            max_size=3,  # Max instances
            min_size=1,  # Min instances (set to 0 for scale-to-zero)
        )
        return auto_scaling_config.attr_auto_scaling_configuration_arn
