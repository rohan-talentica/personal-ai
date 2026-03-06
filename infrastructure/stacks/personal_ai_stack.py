"""
Personal AI Stack
=================
Deploys the Personal AI Research Assistant to AWS ECS Fargate.

Architecture
------------
    Internet ──→ ALB (port 80) ──→ ECS Fargate Task (personal-ai-api)
                                          │
                                    ContainerImage.from_asset()
                                      → CDK builds Dockerfile locally
                                      → CDK pushes to a managed ECR repo
                                    SSM Parameters (OPENROUTER_API_KEY)
                                    CloudWatch Logs (/ecs/personal-ai-api)

Design decisions
----------------
- ContainerImage.from_asset()        → CDK builds + pushes Docker image automatically
- Public subnets only (no NAT gw)   → saves ~$32/month
- Ephemeral local disk for ChromaDB → data resets on task restart (fine for learning)
- 0.5 vCPU / 1 GB RAM Fargate task  → cheapest tier that runs Python ML libs
- Single desired_count=1             → no HA, keeps costs minimal
- SSM StringParameter for secrets    → never bake API keys into the image
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_iam as iam,
    aws_logs as logs,
    aws_ssm as ssm,
)
from constructs import Construct


class PersonalAiStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        openrouter_api_key: str,
        groq_api_key: str,
        database_url: str,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── 1. VPC ───────────────────────────────────────────────────────────
        # Public-only VPC (2 AZs, no NAT gateway) to keep the bill low.
        # Fargate gets a public IP so it can reach ECR & OpenRouter directly.
        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                )
            ],
        )

        # ── 2. ECS Cluster ───────────────────────────────────────────────────
        cluster = ecs.Cluster(
            self,
            "Cluster",
            vpc=vpc,
            cluster_name="personal-ai-cluster",
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )

        # ── 3. CloudWatch Log Group ──────────────────────────────────────────
        log_group = logs.LogGroup(
            self,
            "LogGroup",
            log_group_name="/ecs/personal-ai-api",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ── 4. SSM Parameters for Secrets ────────────────────────────────────
        openrouter_param = ssm.StringParameter(
            self,
            "OpenRouterApiKey",
            parameter_name="/personal-ai/OPENROUTER_API_KEY",
            string_value=openrouter_api_key,
            description="OpenRouter API key — update via CLI before deploying",
            tier=ssm.ParameterTier.STANDARD,
        )
        groq_param = ssm.StringParameter(
            self,
            "GroqApiKey",
            parameter_name="/personal-ai/GROQ_API_KEY",
            string_value=groq_api_key,
            description="Groq API key — update via CLI before deploying",
            tier=ssm.ParameterTier.STANDARD,
        )
        db_param = ssm.StringParameter(
            self,
            "DatabaseUrl",
            parameter_name="/personal-ai/DATABASE_URL",
            string_value=database_url,
            description="Supabase PgVector Database URL — update via CLI before deploying",
            tier=ssm.ParameterTier.STANDARD,
        )

        # ── 5. IAM Roles ─────────────────────────────────────────────────────
        # Execution role: used by ECS agent to pull images & push logs
        execution_role = iam.Role(
            self,
            "EcsExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )
        # Allow execution role to read the SSM params at container start-up
        openrouter_param.grant_read(execution_role)
        groq_param.grant_read(execution_role)
        db_param.grant_read(execution_role)

        # Task role: runtime permissions for the container process itself
        task_role = iam.Role(
            self,
            "EcsTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        # ── 6. Fargate Task Definition ───────────────────────────────────────
        # 0.5 vCPU / 1 GB — smallest size that comfortably runs
        # sentence-transformers + ChromaDB + FastAPI simultaneously.
        task_definition = ecs.FargateTaskDefinition(
            self,
            "TaskDef",
            cpu=512,
            memory_limit_mib=1024,
            execution_role=execution_role,
            task_role=task_role,
        )

        # ── 7. Container Image (CDK-managed) ─────────────────────────────────
        # CDK builds the Dockerfile at ../Dockerfile (project root),
        # creates a CDK-managed ECR repo, and pushes the image automatically
        # during `cdk deploy`. No manual docker build/push needed.
        #
        # LINUX_AMD64 is explicit so the image works on Fargate even when
        # built on an Apple Silicon (ARM) Mac.
        image = ecs.ContainerImage.from_asset(
            "..",                                   # project root (relative to infrastructure/)
            file="Dockerfile",
            build_args={"target": "runtime"},
            platform=ecr_assets.Platform.LINUX_AMD64,
        )

        container = task_definition.add_container(
            "ApiContainer",
            container_name="personal-ai-api",
            image=image,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="api",
                log_group=log_group,
            ),
            environment={
                "PORT": "8000",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            secrets={
                "OPENROUTER_API_KEY": ecs.Secret.from_ssm_parameter(openrouter_param),
                "GROQ_API_KEY": ecs.Secret.from_ssm_parameter(groq_param),
                "DATABASE_URL": ecs.Secret.from_ssm_parameter(db_param),
            },
            health_check=ecs.HealthCheck(
                command=[
                    "CMD-SHELL",
                    (
                        "python -c \""
                        "import urllib.request; "
                        "urllib.request.urlopen('http://localhost:8000/health')"
                        "\" || exit 1"
                    ),
                ],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                retries=3,
                start_period=Duration.seconds(60),
            ),
        )

        container.add_port_mappings(ecs.PortMapping(container_port=8000))

        # ── 8. ALB + Fargate Service ─────────────────────────────────────────
        # ApplicationLoadBalancedFargateService wires up ALB, listener,
        # target group, security groups, and the ECS service in one construct.
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "FargateService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=1,
            public_load_balancer=True,
            assign_public_ip=True,           # Required: no NAT gateway
            service_name="personal-ai-service",
            listener_port=80,
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            health_check_grace_period=Duration.seconds(90),
        )

        # Fine-tune ALB target group health check
        fargate_service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        # Allow inbound on port 8000 from the ALB's security group
        fargate_service.service.connections.security_groups[0].add_ingress_rule(
            peer=fargate_service.load_balancer.connections.security_groups[0],
            connection=ec2.Port.tcp(8000),
            description="ALB → Fargate task port 8000",
        )

        # ── 9. Outputs ───────────────────────────────────────────────────────
        CfnOutput(
            self,
            "ApiUrl",
            value=f"http://{fargate_service.load_balancer.load_balancer_dns_name}",
            description="Public URL of the Personal AI API",
            export_name="PersonalAiApiUrl",
        )
        CfnOutput(
            self,
            "EcsClusterName",
            value=cluster.cluster_name,
            description="ECS Cluster name",
        )
