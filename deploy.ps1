# Quick Deploy Script for Azure
# Run this script to deploy both backend and frontend to Azure

param(
    [string]$Environment = "production",
    [string]$Location = "eastus",
    [switch]$SkipDockerBuild = $false
)

# Configuration
$resourceGroup = "smart-maintenance-rg"
$registryName = "smartmanufacturingacr"
$backendAppName = "smart-maintenance-api-meghna"
$frontendAppName = "smart-manufacturing-frontend"
$registryUrl = "$registryName.azurecr.io"

Write-Host "🚀 Smart Manufacturing GenAI - Azure Deployment Script" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host ""

# Step 1: Check Prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Blue
$azCli = az --version 2>$null
if (-not $azCli) {
    Write-Host "❌ Azure CLI not found. Install from https://learn.microsoft.com/en-us/cli/azure/" -ForegroundColor Red
    exit 1
}

$docker = docker --version 2>$null
if (-not $docker) {
    Write-Host "❌ Docker not found. Install from https://www.docker.com/" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Prerequisites met" -ForegroundColor Green
Write-Host ""

# Step 2: Login to Azure
Write-Host "🔐 Logging into Azure..." -ForegroundColor Blue
az login --output none
Write-Host "✅ Azure login successful" -ForegroundColor Green
Write-Host ""

# Step 3: Build Docker Images
if (-not $SkipDockerBuild) {
    Write-Host "🐳 Building Docker images..." -ForegroundColor Blue
    
    Write-Host "  Building backend image..." -ForegroundColor Yellow
    Push-Location Backend
    docker build -t "$registryUrl/smart-manufacturing-api:latest" -t "$registryUrl/smart-manufacturing-api:$($env:GITHUB_SHA.Substring(0,8))" . 2>&1 | Select-Object -Last 1
    Pop-Location
    
    Write-Host "  Building frontend image..." -ForegroundColor Yellow
    docker build -t "$registryUrl/smart-manufacturing-frontend:latest" -t "$registryUrl/smart-manufacturing-frontend:$($env:GITHUB_SHA.Substring(0,8))" -f Frontend.Dockerfile . 2>&1 | Select-Object -Last 1
    
    Write-Host "✅ Docker images built" -ForegroundColor Green
    Write-Host ""
}

# Step 4: Login to Container Registry
Write-Host "🔑 Logging into Container Registry..." -ForegroundColor Blue
az acr login --name $registryName --output none
Write-Host "✅ Container Registry login successful" -ForegroundColor Green
Write-Host ""

# Step 5: Push Docker Images
Write-Host "📤 Pushing Docker images to registry..." -ForegroundColor Blue
Write-Host "  Pushing backend image..." -ForegroundColor Yellow
docker push "$registryUrl/smart-manufacturing-api:latest" | Select-Object -Last 1
Write-Host "  Pushing frontend image..." -ForegroundColor Yellow
docker push "$registryUrl/smart-manufacturing-frontend:latest" | Select-Object -Last 1
Write-Host "✅ Docker images pushed" -ForegroundColor Green
Write-Host ""

# Step 6: Get Registry Credentials
Write-Host "🔐 Retrieving registry credentials..." -ForegroundColor Blue
$registryPassword = az acr credential show `
    --resource-group $resourceGroup `
    --name $registryName `
    --query passwords[0].value -o tsv

$registryUsername = az acr credential show `
    --resource-group $resourceGroup `
    --name $registryName `
    --query username -o tsv

Write-Host "✅ Credentials retrieved" -ForegroundColor Green
Write-Host ""

# Step 7: Update App Service Container Configuration
Write-Host "⚙️  Configuring App Services..." -ForegroundColor Blue

Write-Host "  Configuring backend..." -ForegroundColor Yellow
az webapp config container set `
    --name $backendAppName `
    --resource-group $resourceGroup `
    --docker-custom-image-name "$registryUrl/smart-manufacturing-api:latest" `
    --docker-registry-server-url "https://$registryUrl" `
    --docker-registry-server-user $registryUsername `
    --docker-registry-server-password $registryPassword `
    --output none

Write-Host "  Configuring frontend..." -ForegroundColor Yellow
az webapp config container set `
    --name $frontendAppName `
    --resource-group $resourceGroup `
    --docker-custom-image-name "$registryUrl/smart-manufacturing-frontend:latest" `
    --docker-registry-server-url "https://$registryUrl" `
    --docker-registry-server-user $registryUsername `
    --docker-registry-server-password $registryPassword `
    --output none

Write-Host "✅ App Services configured" -ForegroundColor Green
Write-Host ""

# Step 8: Configure Environment Variables
Write-Host "🔧 Setting application settings..." -ForegroundColor Blue

az webapp config appsettings set `
    --name $backendAppName `
    --resource-group $resourceGroup `
    --settings `
    WEBSITES_ENABLE_APP_SERVICE_STORAGE=false `
    FLASK_ENV=$Environment `
    PYTHONUNBUFFERED=1 `
    WEBSITES_PORT=8000 `
    --output none

az webapp config appsettings set `
    --name $frontendAppName `
    --resource-group $resourceGroup `
    --settings `
    WEBSITES_ENABLE_APP_SERVICE_STORAGE=false `
    WEBSITES_PORT=3000 `
    REACT_APP_API_URL="https://$backendAppName.azurewebsites.net" `
    --output none

Write-Host "✅ Application settings configured" -ForegroundColor Green
Write-Host ""

# Step 9: Restart App Services
Write-Host "🔄 Restarting App Services..." -ForegroundColor Blue
az webapp restart --name $backendAppName --resource-group $resourceGroup --output none
az webapp restart --name $frontendAppName --resource-group $resourceGroup --output none
Write-Host "✅ App Services restarted" -ForegroundColor Green
Write-Host ""

# Step 10: Display Results
Write-Host "✨ Deployment Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Access your applications at:" -ForegroundColor Cyan
Write-Host "  Backend API: https://$backendAppName.azurewebsites.net" -ForegroundColor Yellow
Write-Host "  Frontend:    https://$frontendAppName.azurewebsites.net" -ForegroundColor Yellow
Write-Host ""
Write-Host "📊 View logs:" -ForegroundColor Cyan
Write-Host "  az webapp log tail --name $backendAppName --resource-group $resourceGroup" -ForegroundColor Yellow
Write-Host "  az webapp log tail --name $frontendAppName --resource-group $resourceGroup" -ForegroundColor Yellow
Write-Host ""
Write-Host "🎯 Next steps:" -ForegroundColor Cyan
Write-Host "  1. Test the backend API endpoint" -ForegroundColor Gray
Write-Host "  2. Configure custom domain (optional)" -ForegroundColor Gray
Write-Host "  3. Set up Application Insights for monitoring" -ForegroundColor Gray
Write-Host "  4. Configure GitHub Actions for CI/CD" -ForegroundColor Gray
