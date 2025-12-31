# Azure Deployment Setup

## 📋 Quick Start

Your Smart Manufacturing GenAI app is ready for Azure deployment! Here's what has been created:

### Files Created:
- **`deploy.ps1`** - One-command deployment script (recommended)
- **`Backend/Dockerfile`** - Container image for Flask API
- **`Frontend.Dockerfile`** - Container image for React frontend
- **`docker-compose.yml`** - Local development setup
- **`.github/workflows/azure-deploy.yml`** - CI/CD automation
- **`azure-deploy-guide.md`** - Detailed deployment documentation

---

## 🚀 Deploy in 3 Steps

### Step 1: Install Prerequisites (One-time)
```powershell
# Install Azure CLI
# Download from: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli

# Install Docker
# Download from: https://www.docker.com/products/docker-desktop
```

### Step 2: Run the Deployment Script
```powershell
# Navigate to the project root directory
cd "c:\Users\meghn\Downloads\SMART MANUFACTURING GENAI"

# Run the deployment script
.\deploy.ps1
```

### Step 3: Access Your App
```
Backend API: https://smart-maintenance-api-meghna.azurewebsites.net
Frontend:    https://smart-manufacturing-frontend.azurewebsites.net
```

---

## 🔧 What the Script Does

The `deploy.ps1` script automates:
1. ✅ Checks prerequisites (Azure CLI, Docker)
2. ✅ Authenticates with Azure
3. ✅ Builds Docker images for backend and frontend
4. ✅ Pushes images to Azure Container Registry
5. ✅ Updates App Service configurations
6. ✅ Sets environment variables
7. ✅ Restarts services

**Estimated time: 10-15 minutes**

---

## 📦 Architecture

```
┌─────────────────────────────────────────────┐
│        Azure Container Registry              │
├─────────────────────────────────────────────┤
│  smart-manufacturing-api:latest              │
│  smart-manufacturing-frontend:latest         │
└─────────────────────────────────────────────┘
              ↓                    ↓
┌──────────────────────────────────────────────┐
│         Azure App Service (Linux)             │
├──────────────────────────────────────────────┤
│ Backend API (Port 8000)  Frontend (Port 3000)│
└──────────────────────────────────────────────┘
```

---

## 🛠️ Manual Deployment (Alternative)

If you prefer manual control, see `azure-deploy-guide.md` for step-by-step instructions.

---

## 📊 Monitor Your App

### View Logs
```powershell
# Backend logs
az webapp log tail --name smart-maintenance-api-meghna --resource-group smart-maintenance-rg

# Frontend logs
az webapp log tail --name smart-manufacturing-frontend --resource-group smart-maintenance-rg
```

### Check Health
```powershell
# Test backend
curl https://smart-maintenance-api-meghna.azurewebsites.net/

# Test frontend (should return HTML)
curl https://smart-manufacturing-frontend.azurewebsites.net
```

---

## 🔄 Update Your App

After making code changes, redeploy with:
```powershell
.\deploy.ps1
```

---

## 🔒 Security Best Practices

1. **Store secrets in Azure Key Vault**
   ```powershell
   az keyvault create --name smart-manufacturing-kv --resource-group smart-maintenance-rg
   ```

2. **Enable HTTPS** (Azure provides SSL by default)

3. **Use environment variables** for sensitive data
   ```powershell
   az webapp config appsettings set --name smart-maintenance-api-meghna \
       --resource-group smart-maintenance-rg \
       --settings OPENAI_API_KEY="your-key-here"
   ```

4. **Enable Application Insights** for monitoring
   ```powershell
   az monitor app-insights component create \
       --app smart-manufacturing-insights \
       --location eastus \
       --resource-group smart-maintenance-rg \
       --application-type web
   ```

---

## 💾 Database & Storage (Future)

If you need a database:
```powershell
# Create Azure SQL Database
az sql server create --name smart-manufacturing-sql \
    --resource-group smart-maintenance-rg \
    --admin-user sqladmin \
    --admin-password YourSecurePassword123

# Or use Azure Blob Storage for files
az storage account create --name smartmfgstorage \
    --resource-group smart-maintenance-rg
```

---

## 🧪 Local Development Testing

Test locally before deploying:
```powershell
# Start both frontend and backend with Docker Compose
docker-compose up

# Access locally:
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

---

## ❓ Troubleshooting

### App won't start
```powershell
# Check logs first
az webapp log tail --name smart-maintenance-api-meghna --resource-group smart-maintenance-rg

# Restart the app
az webapp restart --name smart-maintenance-api-meghna --resource-group smart-maintenance-rg
```

### Image not found error
```powershell
# Verify image was pushed successfully
az acr repository list --name smartmanufacturingacr
```

### CORS errors
- Backend already has CORS enabled
- Verify frontend is using correct API URL in environment variables

### Timeout errors
- Flask/Gunicorn timeout is set to 600 seconds
- Increase if training/prediction takes longer:
  ```powershell
  az webapp config appsettings set --name smart-maintenance-api-meghna \
      --resource-group smart-maintenance-rg \
      --settings WEBSITES_ENABLE_APP_SERVICE_STORAGE=false
  ```

---

## 📞 Support Resources

- **Azure CLI Documentation**: https://learn.microsoft.com/en-us/cli/azure/
- **Azure App Service**: https://learn.microsoft.com/en-us/azure/app-service/
- **Docker Documentation**: https://docs.docker.com/
- **GitHub Actions**: https://docs.github.com/en/actions

---

## ✨ Next Steps

1. ✅ Deploy with `deploy.ps1`
2. ✅ Test the application
3. ⬜ Set up CI/CD with GitHub Actions (see azure-deploy-guide.md)
4. ⬜ Configure custom domain
5. ⬜ Enable Application Insights
6. ⬜ Set up automated backups

---

**Happy Deploying! 🎉**
