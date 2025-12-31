# 🏭 Smart Manufacturing GenAI Assistant

An end-to-end **AI-powered smart manufacturing platform** that predicts machine failures, explains risks, recommends actions, and enables what-if simulations — deployed as a **cloud-native containerized application**.

This project demonstrates **real-world AI engineering**, combining **machine learning, explainable AI, backend systems, Docker, and cloud deployment**.

---

## 🎯 Problem Statement

Modern manufacturing systems generate massive sensor data, yet most predictive systems only show *risk scores* without explaining **why failures happen** or **what actions should be taken**.

This project solves that gap by providing:
- Predictive maintenance
- Explainability for predictions
- Actionable maintenance recommendations
- What-if simulations
- Cloud-deployed AI backend

---

## 🌐 Cloud Deployment (Docker + Azure Container Apps)

The backend of this project is deployed as a **containerized AI service** using **Docker and Azure Container Apps**.

### High-Level Architecture

React Frontend (Browser)
|
| HTTPS REST API
↓
Azure Container Apps
(FastAPI + ML Inference Engine)
|
↓
Trained ML Models (PyTorch / Scikit-learn)

markdown
Copy code

### Why this matters
- No local setup required for users
- Production-grade AI deployment
- Auto-scaling serverless containers
- Real-world DevOps + ML integration

---

## 🚀 Key Features

### 🔮 Predictive Maintenance
- ML models predict failure probability for machines
- Risk scores generated in real time

### 🧠 Explainable AI
- Feature-level explanation of predictions
- Highlights top contributing sensor parameters

### 🛠️ Action Engine
- Rule-based + ML-driven maintenance recommendations
- Converts AI insights into **real operational actions**

### 🔁 What-If Simulation
- Modify sensor inputs
- Instantly observe impact on failure risk

### 🌐 Cloud-Hosted AI API
- Backend deployed via Docker
- Public HTTPS endpoint
- Swagger API documentation enabled

---

## 🧰 Tech Stack

### Backend
- **Python 3.10**
- **FastAPI**
- **Gunicorn + Uvicorn**
- **PyTorch**
- **Scikit-learn**
- **Pandas / NumPy**
- **OpenAI API (GenAI-ready)**

### Frontend
- **React**
- **Vite**
- **Modern UI Components**

### DevOps & Cloud
- **Docker**
- **Azure Container Apps**
- **Docker Hub**
- **GitHub**

---

## 📁 Project Structure

SMART-MANUFACTURING-GEN-AI/
│
├── Backend/
│ ├── app/
│ │ ├── api/
│ │ ├── ml/
│ │ └── services/
│ ├── Dockerfile
│ ├── .dockerignore
│ ├── requirements.txt
│ └── requirements-runtime.txt
│
├── webapp/
│ ├── src/
│ └── public/
│
└── README.md

yaml
Copy code

---

## 🐳 Dockerization Details

The backend is fully containerized for reproducible deployment.

### Key Files
Backend/
├── Dockerfile
├── .dockerignore
├── requirements-runtime.txt

shell
Copy code

### Docker Image
meghna0507/sm-backend:latest

yaml
Copy code

### Container Configuration
- Runtime: Python 3.10
- Server: Gunicorn + Uvicorn
- Exposed Port: 8000
- CPU-only inference (cost efficient)

### Why Docker?
- Environment consistency
- Cloud portability
- Faster deployments
- No “works on my machine” issues

---

## ☁️ Cloud Deployment (Recommended)

### Backend – Azure Container Apps
- Serverless container execution
- Auto HTTPS & ingress
- Automatic scaling
- No Kubernetes configuration required

**API Endpoint:**
https://<azure-container-app-url>

markdown
Copy code

**Swagger Docs:**
https://<azure-container-app-url>/docs

yaml
Copy code

---

## 🧪 Running Locally (Optional)

### Backend
```bash
cd Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
Frontend
bash
Copy code
cd webapp
npm install
npm run dev
☁️ Why Azure Container Apps (Design Choice)
Azure Container Apps was chosen because it offers:

Serverless container hosting

Built-in HTTPS

Auto-scaling

Low-cost / free-tier friendly

Ideal for ML inference APIs

This allows focusing on AI logic instead of infrastructure management.

🧠 Learning Outcomes
End-to-end AI system design

Explainable ML in production

Docker-based ML deployment

Cloud-native AI backend

Frontend–backend integration

Real-world DevOps exposure

📝 Future Enhancements
SHAP-based deep explainability

Real-time sensor streaming

GenAI-powered maintenance chatbot

Multi-factory dashboard

Cost optimization with smaller base images

👩‍💻 Author
Meghna Soni
AI / ML Engineer | Smart Manufacturing | GenAI Systems

Built with PyTorch, FastAPI, React, Docker, and Azure Container Apps