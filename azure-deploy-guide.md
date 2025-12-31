# Azure Deployment Guide for Smart Manufacturing GenAI

## Prerequisites

1. **Azure Account** ✓ (you have this)
2. **Azure CLI** - Download from https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
3. **Docker** - Download from https://www.docker.com/products/docker-desktop
4. **Git** (for CI/CD)

---

## Step 1: Set Up Azure Resources

### 1.1 Login to Azure
```powershell
az login
```

### 1.2 Create Resource Group (if not exists)
```powershell
az group create --name smart-maintenance-rg --location eastus
```

### 1.3 Create Container Registry
```powershell
az acr create --resource-group smart-maintenance-rg `
              --name smartmanufacturingacr `
              --sku Basic
```

### 1.4 Create App Service Plans
```powershell
# For Backend API
az appservice plan create --name ASP-smartmaintenancerg-88e1 `
                          --resource-group smart-maintenance-rg `
                          --is-linux `
                          --sku B2

# For Frontend
az appservice plan create --name ASP-smartmfg-frontend `
                          --resource-group smart-maintenance-rg `
                          --is-linux `
                          --sku B2
```

### 1.5 Create Web Apps
```powershell
# Backend API
az webapp create --resource-group smart-maintenance-rg `
                 --plan ASP-smartmaintenancerg-88e1 `
                 --name smart-maintenance-api-meghna `
                 --deployment-container-image-name-user smartmanufacturingacr.azurecr.io/smart-manufacturing-api:latest

# Frontend
az webapp create --resource-group smart-maintenance-rg `
                 --plan ASP-smartmfg-frontend `
                 --name smart-manufacturing-frontend `
                 --deployment-container-image-name-user smartmanufacturingacr.azurecr.io/smart-manufacturing-frontend:latest
```

---

## Step 2: Configure Container Registry Access

### 2.1 Get Registry Credentials
```powershell
$acrUrl = az acr show --resource-group smart-maintenance-rg `
                      --name smartmanufacturingacr `
                      --query loginServer --output tsv

$acrUsername = az acr credential show --resource-group smart-maintenance-rg `
                                      --name smartmanufacturingacr `
                                      --query username --output tsv

$acrPassword = az acr credential show --resource-group smart-maintenance-rg `
                                      --name smartmanufacturingacr `
                                      --query passwords[0].value --output tsv

echo "Registry: $acrUrl"
echo "Username: $acrUsername"
echo "Password: $acrPassword"
```

### 2.2 Configure App Service to Access Registry
```powershell
# For Backend
az webapp config container set --name smart-maintenance-api-meghna `
                               --resource-group smart-maintenance-rg `
                               --docker-custom-image-name smartmanufacturingacr.azurecr.io/smart-manufacturing-api:latest `
                               --docker-registry-server-url https://smartmanufacturingacr.azurecr.io `
                               --docker-registry-server-user $acrUsername `
                               --docker-registry-server-password $acrPassword

# For Frontend
az webapp config container set --name smart-manufacturing-frontend `
                               --resource-group smart-maintenance-rg `
                               --docker-custom-image-name smartmanufacturingacr.azurecr.io/smart-manufacturing-frontend:latest `
                               --docker-registry-server-url https://smartmanufacturingacr.azurecr.io `
                               --docker-registry-server-user $acrUsername `
                               --docker-registry-server-password $acrPassword
```

---

## Step 3: Build and Push Docker Images

### 3.1 Login to Docker Registry
```powershell
az acr login --name smartmanufacturingacr
```

### 3.2 Build Backend Image
```powershell
cd Backend
docker build -t smartmanufacturingacr.azurecr.io/smart-manufacturing-api:latest .
docker push smartmanufacturingacr.azurecr.io/smart-manufacturing-api:latest
cd ..
```

### 3.3 Build Frontend Image
```powershell
docker build -t smartmanufacturingacr.azurecr.io/smart-manufacturing-frontend:latest -f Frontend.Dockerfile .
docker push smartmanufacturingacr.azurecr.io/smart-manufacturing-frontend:latest
```

---

## Step 4: Configure Environment Variables

### 4.1 Backend App Settings
```powershell
az webapp config appsettings set --name smart-maintenance-api-meghna `
                                  --resource-group smart-maintenance-rg `
                                  --settings `
                                  WEBSITES_ENABLE_APP_SERVICE_STORAGE=false `
                                  FLASK_ENV=production `
                                  PYTHONUNBUFFERED=1 `
                                  WEBSITES_PORT=8000
```

### 4.2 Frontend App Settings
```powershell
az webapp config appsettings set --name smart-manufacturing-frontend `
                                  --resource-group smart-maintenance-rg `
                                  --settings `
                                  WEBSITES_ENABLE_APP_SERVICE_STORAGE=false `
                                  WEBSITES_PORT=3000 `
                                  REACT_APP_API_URL=https://smart-maintenance-api-meghna.azurewebsites.net
```

---

## Step 5: Deploy Application

### 5.1 Deploy Backend
```powershell
az webapp up --name smart-maintenance-api-meghna `
             --resource-group smart-maintenance-rg `
             --sku B2 `
             --os-type Linux `
             --runtime "DOCKER|smartmanufacturingacr.azurecr.io/smart-manufacturing-api:latest"
```

### 5.2 Deploy Frontend
```powershell
az webapp up --name smart-manufacturing-frontend `
             --resource-group smart-maintenance-rg `
             --sku B2 `
             --os-type Linux `
             --runtime "DOCKER|smartmanufacturingacr.azurecr.io/smart-manufacturing-frontend:latest"
```

---

## Step 6: Verify Deployment

### 6.1 Check Backend
```powershell
curl https://smart-maintenance-api-meghna.azurewebsites.net/
```

### 6.2 Check Frontend
```powershell
# Open in browser
https://smart-manufacturing-frontend.azurewebsites.net
```

### 6.3 View Logs
```powershell
az webapp log tail --name smart-maintenance-api-meghna `
                   --resource-group smart-maintenance-rg

az webapp log tail --name smart-manufacturing-frontend `
                   --resource-group smart-maintenance-rg
```

---

## Step 7: Set Up CI/CD (Optional but Recommended)

### 7.1 Store Secrets in GitHub
Go to your GitHub repository → Settings → Secrets and add:
- `AZURE_REGISTRY_LOGIN_SERVER`: smartmanufacturingacr.azurecr.io
- `AZURE_REGISTRY_USERNAME`: (from step 2.1)
- `AZURE_REGISTRY_PASSWORD`: (from step 2.1)
- `AZURE_PUBLISH_PROFILE_BACKEND`: (download from Azure Portal)
- `AZURE_PUBLISH_PROFILE_FRONTEND`: (download from Azure Portal)

### 7.2 Push GitHub Actions Workflow
The `.github/workflows/azure-deploy.yml` file will automatically deploy on push to main branch.

---

## Troubleshooting

### Issue: Container won't start
**Solution:**
```powershell
az webapp log tail --name smart-maintenance-api-meghna --resource-group smart-maintenance-rg
```
Check the logs for errors.

### Issue: Model files not found
**Solution:** 
Mount Azure Storage or add models to the Docker image. Update Dockerfile to copy model artifacts:
```dockerfile
COPY app/models_artifacts/ /app/models_artifacts/
```

### Issue: CORS errors
**Solution:**
Backend already has CORS enabled. Make sure frontend uses the correct API URL:
```javascript
// In frontend/.env
REACT_APP_API_URL=https://smart-maintenance-api-meghna.azurewebsites.net
```

---

## Next Steps

1. ✅ Create Azure resources
2. ✅ Build and push Docker images
3. ✅ Configure environment variables
4. ✅ Deploy applications
5. ✅ Set up CI/CD for automatic deployments
6. Monitor performance and costs in Azure Portal
7. Set up Application Insights for monitoring

---

## Quick Deploy Script (All-in-One)

Save this as `deploy.ps1` and run it:

```powershell
# Configuration
$resourceGroup = "smart-maintenance-rg"
$location = "eastus"
$registryName = "smartmanufacturingacr"
$backendAppName = "smart-maintenance-api-meghna"
$frontendAppName = "smart-manufacturing-frontend"

# Login
az login

# Create resources (if needed)
az group create --name $resourceGroup --location $location

# Build and push images
az acr login --name $registryName
docker build -t $registryName.azurecr.io/smart-manufacturing-api:latest -f Backend/Dockerfile .
docker push $registryName.azurecr.io/smart-manufacturing-api:latest
docker build -t $registryName.azurecr.io/smart-manufacturing-frontend:latest -f Frontend.Dockerfile .
docker push $registryName.azurecr.io/smart-manufacturing-frontend:latest

# Update web apps
az webapp config container set --name $backendAppName `
                               --resource-group $resourceGroup `
                               --docker-custom-image-name $registryName.azurecr.io/smart-manufacturing-api:latest `
                               --docker-registry-server-url https://$registryName.azurecr.io

az webapp config container set --name $frontendAppName `
                               --resource-group $resourceGroup `
                               --docker-custom-image-name $registryName.azurecr.io/smart-manufacturing-frontend:latest `
                               --docker-registry-server-url https://$registryName.azurecr.io

echo "✅ Deployment complete!"
echo "Backend: https://$backendAppName.azurewebsites.net"
echo "Frontend: https://$frontendAppName.azurewebsites.net"
```

Run with:
```powershell
.\deploy.ps1
```
