# deploy_lambda.ps1
# ------------------
# Build the Lambda container image, push to ECR, update the Lambda function.
#
# Usage:
#   .\deploy_lambda.ps1
# ---------------------------------------------------------------------------

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Config ──────────────────────────────────────────────────────────────────
$AWS_REGION      = "eu-north-1"
$AWS_ACCOUNT_ID  = (& "C:\Program Files\Amazon\AWSCLIV2\aws.exe" sts get-caller-identity --query Account --output text --region $AWS_REGION)
$ECR_REPO        = "gov-schemes-lambda"
$LAMBDA_NAME     = "gov-schemes-voice-rag"
$IMAGE_TAG       = "latest"
$ECR_URI         = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
$AWS             = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$DOCKER          = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"

$env:PATH += ";C:\Program Files\Docker\Docker\resources\bin"

Write-Host ""
Write-Host "=" * 60
Write-Host "  Lambda Deploy: $LAMBDA_NAME"
Write-Host "  ECR           : $ECR_URI"
Write-Host "  Region        : $AWS_REGION"
Write-Host "=" * 60
Write-Host ""

# ── Step 1: ECR login ────────────────────────────────────────────────────────
Write-Host "[1/4] Logging into ECR..."
$login = & $AWS ecr get-login-password --region $AWS_REGION
$login | & $DOCKER login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
Write-Host "      OK"

# ── Step 2: Build image ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/4] Building Docker image (this takes ~5 min first time, ~1 min on rebuild)..."
& $DOCKER build --platform linux/amd64 -f Dockerfile.lambda -t "${ECR_REPO}:${IMAGE_TAG}" .
if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }
Write-Host "      Build complete."

# ── Step 3: Tag + push ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/4] Tagging and pushing to ECR..."
& $DOCKER tag "${ECR_REPO}:${IMAGE_TAG}" $ECR_URI
& $DOCKER push $ECR_URI
if ($LASTEXITCODE -ne 0) { throw "Docker push failed" }
Write-Host "      Push complete."

# ── Step 4: Update Lambda ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "[4/4] Updating Lambda function image..."
& $AWS lambda update-function-code `
    --function-name $LAMBDA_NAME `
    --image-uri $ECR_URI `
    --region $AWS_REGION `
    --output json | ConvertFrom-Json | Select-Object FunctionName, LastModified, CodeSize

# Wait for update to complete
Write-Host "      Waiting for Lambda update to propagate..."
& $AWS lambda wait function-updated --function-name $LAMBDA_NAME --region $AWS_REGION
Write-Host "      Lambda is live."

Write-Host ""
Write-Host "=" * 60
Write-Host "  Deploy complete!"
Write-Host "  Lambda: $LAMBDA_NAME @ $AWS_REGION"
Write-Host "  Image : $ECR_URI"
Write-Host "=" * 60
Write-Host ""
