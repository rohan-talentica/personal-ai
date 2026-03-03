#!/usr/bin/env python3
import os
from pathlib import Path

import aws_cdk as cdk
from dotenv import load_dotenv

from stacks.personal_ai_stack import PersonalAiStack

# Load secrets from the project-root .env (one level above infrastructure/)
load_dotenv(Path(__file__).parent.parent / ".env")

app = cdk.App()

PersonalAiStack(
    app,
    "PersonalAiStack",
    openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
    env=cdk.Environment(
        account="044079590862",
        region="ap-south-1",
    ),
    description="Personal AI Research Assistant",
)

app.synth()
