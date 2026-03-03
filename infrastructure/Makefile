# ─────────────────────────────────────────────────────────────────────────────
# Personal AI — Deployment Makefile
# Run all commands from the infrastructure/ directory.
#
# CDK builds + pushes the Docker image automatically during `cdk deploy`
# via ContainerImage.from_asset() — no manual docker push needed.
# ─────────────────────────────────────────────────────────────────────────────

AWS_PROFILE := personal-ai
AWS_REGION  := ap-south-1
AWS_ACCOUNT := 044079590862

.PHONY: help install bootstrap synth push-secret deploy destroy logs status

## ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Personal AI – Deployment Commands"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make install       Install CDK Python deps (run once)"
	@echo "  make bootstrap     Bootstrap CDK in AWS account (run once)"
	@echo "  make synth         Preview the CloudFormation template"
	@echo "  make push-secret   Store OPENROUTER_API_KEY in SSM"
	@echo "  make deploy        Build image + deploy everything to AWS"
	@echo "  make destroy       Tear down ALL AWS resources"
	@echo "  make logs          Tail ECS container logs"
	@echo "  make status        Show ECS service status"
	@echo ""
	@echo "  Full first-time flow:"
	@echo "    make install → make bootstrap → make push-secret KEY=sk-or-... → make deploy"
	@echo ""

## ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

bootstrap:
	cdk bootstrap \
	  aws://$(AWS_ACCOUNT)/$(AWS_REGION) \
	  --profile $(AWS_PROFILE)

## ── CDK ──────────────────────────────────────────────────────────────────────
synth:
	cdk synth --profile $(AWS_PROFILE)

## ── Secrets (run BEFORE deploy) ──────────────────────────────────────────────
# Usage:  make push-secret KEY=sk-or-v1-xxxxx
push-secret:
	@if [ -z "$(KEY)" ]; then \
	  echo "ERROR: Provide your key with: make push-secret KEY=sk-or-..."; exit 1; \
	fi
	aws ssm put-parameter \
	  --name "/personal-ai/OPENROUTER_API_KEY" \
	  --value "$(KEY)" \
	  --type SecureString \
	  --overwrite \
	  --profile $(AWS_PROFILE) \
	  --region $(AWS_REGION)
	@echo "✅ Secret stored in SSM"

## ── Deploy ───────────────────────────────────────────────────────────────────
# CDK will:
#   1. Build the Docker image from ../Dockerfile (project root)
#   2. Push the image to a CDK-managed ECR repo
#   3. Create/update all AWS resources (VPC, ECS, ALB, etc.)
deploy:
	cdk deploy \
	  --require-approval never \
	  --profile $(AWS_PROFILE)
	@echo ""
	@echo "✅ Deployment complete! Check the ApiUrl output above."

## ── Redeploy after code changes ─────────────────────────────────────────────
# CDK detects the Dockerfile changed (content hash), rebuilds and redeploys.
redeploy: deploy

## ── Teardown ─────────────────────────────────────────────────────────────────
destroy:
	@echo "⚠️  This will delete ALL resources (VPC, ECS, ALB, ECR, logs)."
	@read -p "Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ]
	cdk destroy --force --profile $(AWS_PROFILE)

## ── Observability ────────────────────────────────────────────────────────────
logs:
	aws logs tail /ecs/personal-ai-api \
	  --follow \
	  --profile $(AWS_PROFILE) \
	  --region $(AWS_REGION)

status:
	aws ecs describe-services \
	  --cluster personal-ai-cluster \
	  --services personal-ai-service \
	  --profile $(AWS_PROFILE) \
	  --region $(AWS_REGION) \
	  --query "services[0].{Status:status,Running:runningCount,Desired:desiredCount}" \
	  --output table
