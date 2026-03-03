# Day 10 — AWS Deployment Guide
## Personal AI Research Assistant → ECS Fargate (ap-south-1)

---

## 🏗️ Architecture

```
Internet ──→ ALB (port 80) ──→ ECS Fargate Task
                                      │
                            ContainerImage.from_asset()
                              → CDK builds Dockerfile locally
                              → CDK pushes to CDK-managed ECR repo
                            SSM Parameter (OPENROUTER_API_KEY)
                            CloudWatch Logs (/ecs/personal-ai-api)
                            Ephemeral disk (ChromaDB @ /app/data/chroma)
```

**Key decisions:**
| Choice | Reasoning |
|---|---|
| `ContainerImage.from_asset()` | CDK builds + pushes Docker image automatically — no manual `docker push` |
| Public subnets, no NAT | Saves ~$32/month. Fargate gets a public IP to reach ECR & OpenRouter. |
| Ephemeral disk for Chroma | No EFS cost, simpler setup. Data resets on task restart — fine for learning. |
| 0.5 vCPU / 1 GB RAM | Smallest tier that runs sentence-transformers + FastAPI comfortably. |
| `desired_count=1` | No HA needed. Keeps the bill minimal. |
| SSM StringParameter | API keys injected at runtime — never baked into the image. |

---

## 📋 Prerequisites

| Tool | Check |
|---|---|
| AWS CLI | `aws --version` |
| AWS CDK v2 | `cdk --version` |
| Docker Desktop | `docker info` (must be running for image build) |
| Python 3.11+ | `python3 --version` |

AWS Profile `personal-ai` → account `044079590862`, region `ap-south-1` ✅

---

## 🚀 Step-by-Step Deployment

All commands run from the `infrastructure/` directory.

### Step 1 — Install CDK Python dependencies
```bash
cd infrastructure/
make install
# or: pip install -r requirements.txt
```

### Step 2 — Bootstrap CDK (one-time per account/region)
Creates an S3 bucket and IAM roles that CDK uses internally.
```bash
make bootstrap
# or:
cdk bootstrap aws://044079590862/ap-south-1 --profile personal-ai
```

### Step 3 — Preview the CloudFormation template
```bash
make synth
# Prints the full CFN template. No AWS resources created yet.
# Look for the Metadata section to confirm Docker image asset is detected.
```

### Step 4 — Store your OpenRouter API key in SSM
```bash
make push-secret KEY=sk-or-v1-your-real-key-here
```

This stores the key in **AWS SSM Parameter Store** as a SecureString at
`/personal-ai/OPENROUTER_API_KEY`. The Fargate task reads it at startup —
the key is never in your Docker image or environment files.

### Step 5 — Deploy 🚀
```bash
make deploy
```

What happens under the hood:
1. CDK computes a **content hash** of your Dockerfile + `src/` directory
2. CDK **builds** the Docker image locally (`docker build --target runtime`)
3. CDK **authenticates** to ECR and **pushes** the image to a CDK-managed repo
4. CDK creates/updates all AWS resources:
   - ✅ VPC (2 public subnets across 2 AZs)
   - ✅ ECS Cluster (`personal-ai-cluster`)
   - ✅ CloudWatch Log Group (`/ecs/personal-ai-api`)
   - ✅ SSM Parameter (already created in Step 4)
   - ✅ IAM Roles (execution + task)
   - ✅ Fargate Task Definition
   - ✅ Application Load Balancer
   - ✅ ECS Fargate Service (`personal-ai-service`)

First deployment takes **~8-12 minutes** (mostly ALB provisioning).

### Step 6 — Test the live API

After `cdk deploy` completes, look for the `ApiUrl` output:
```
Outputs:
PersonalAiStack.ApiUrl = http://Perso-Farga-xxxx.ap-south-1.elb.amazonaws.com
```

```bash
# Health check
curl http://<ApiUrl>/health

# Chat endpoint
curl -X POST http://<ApiUrl>/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangChain?"}'

# Swagger UI
open http://<ApiUrl>/docs
```

---

## 📊 Monitoring

```bash
# Tail live CloudWatch logs
make logs

# Check ECS service running/desired count
make status
```

**AWS Console shortcuts:**
- ECS: `https://ap-south-1.console.aws.amazon.com/ecs/v2/clusters/personal-ai-cluster`
- Logs: `https://ap-south-1.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Fecs$252Fpersonal-ai-api`
- ECR: `https://ap-south-1.console.aws.amazon.com/ecr/repositories`

---

## 🔄 Updating the Application

When you change code or the Dockerfile, just re-run:
```bash
make deploy
```
CDK detects the content hash changed, rebuilds the image, pushes it, and
forces a new ECS deployment automatically.

---

## 💰 Estimated Cost (ap-south-1)

| Resource | ~Monthly Cost |
|---|---|
| ECS Fargate (0.5 vCPU / 1 GB, 24/7) | ~$11 |
| ALB | ~$16 |
| ECR (CDK-managed, < 1 GB) | ~$0.10 |
| CloudWatch Logs (1 week retention) | ~$0.50 |
| SSM Parameters (Standard) | Free |
| **Total** | **~$28/month** |

> 💡 **Tip:** Run `make destroy` when not using it. You can redeploy anytime.

---

## 🧹 Teardown

```bash
make destroy
# Deletes: VPC, ALB, ECS Cluster, ECR repo (+ images), Log Group, SSM params
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `make deploy` fails at Docker build | Ensure Docker Desktop is running |
| `platform` mismatch on Apple Silicon | `LINUX_AMD64` is set explicitly — should be fine |
| Container exits immediately | Run `make logs` to see the Python error |
| Health check failing | Verify `/health` returns 200 locally: `docker compose up` |
| `OPENROUTER_API_KEY` env missing | Re-run `make push-secret KEY=sk-or-...` |
| Task stuck in PROVISIONING | Check security group allows ECR pulls (public IP is required) |

---

*Day 10 Complete — Personal AI Research Assistant deployed to AWS! 🎉*
