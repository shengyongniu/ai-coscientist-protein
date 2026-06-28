#!/usr/bin/env bash
set -euo pipefail
cd /Users/syniu/ai-coscientist
source .venv/bin/activate
export AWS_REGION=us-west-2
export COSCIENTIST_LLM_PROVIDER=bedrock
export COSCIENTIST_MODEL_STRONG=us.anthropic.claude-opus-4-8
export COSCIENTIST_MODEL_FAST=us.anthropic.claude-opus-4-8
mkdir -p data

GOAL="Design improved single-domain antibody (nanobody) variants that bind the SARS-CoV-2 spike RBD with higher predicted stability and affinity than wild-type"

python -m coscientist.cli run "$GOAL" \
  --config protein_binder \
  --provider bedrock \
  --scorer esm \
  --rounds 2 \
  --db data/demo_bedrock.db
