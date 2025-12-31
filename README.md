# Smart Manufacturing GenAI — Predictive Maintenance Decision Engine

## 🎯 Problem Statement

Traditional manufacturing predictive maintenance systems suffer from:
- **Black-box predictions** — ML models output probabilities without explanation
- **Lack of actionability** — Engineers don't know what to do with risk scores
- **Alert fatigue** — Fixed thresholds don't match real operational contexts
- **No feedback loop** — Can't test "what-if" scenarios before acting

**This system bridges the gap** between ML predictions and maintenance decisions.

---

## 🏗️ Architecture: Decision Engine (6-Layer Stack)

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA INGESTION & VALIDATION                        │
│  - CSV upload with schema validation                         │
│  - Categorical encoding (label encoders)                     │
│  - Feature scaling (StandardScaler)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: FAILURE RISK ESTIMATION                            │
│  - PyTorch neural network (19 features → 1 logit)            │
│  - Sigmoid activation → probability [0, 1]                  │
│  - Risk score normalization (0-100)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: EXPLAINABILITY & ATTRIBUTION                       │
│  - Feature contribution scoring (weight × input)             │
│  - Top-3 risk factors extraction                             │
│  - Impact score normalization                                │
│  - [Optional] SHAP-based explanations                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: ACTION MAPPING (Rule-Based)                        │
│  - Priority-driven actions (High/Medium/Low)                 │
│  - Feature-specific domain rules (vibration→bearings, etc)  │
│  - Urgency classification + timeline estimation              │
│  - Delay impact quantification                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5: SIMULATION & OPTIMIZATION                          │
│  - What-if analysis endpoint (/simulate)                     │
│  - Feature modification impact modeling                      │
│  - Real-time risk re-evaluation                              │
│  - Decision support for proactive intervention               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 6: FRONTEND VISUALIZATION & UX                        │
│  - Dashboard with KPI cards (High Risk, Medium Risk, etc)    │
│  - Expandable result rows                                    │
│  - Explainability details (Why? What? When?)                │
│  - Recommended actions with urgency indicators               │
│  - What-if sliders (coming soon)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Features

### 1. **Risk Scoring & Classification**
- **Raw Output**: Probability [0, 1] from trained PyTorch network
- **Normalized Score**: 0-100 integer risk scale
- **Thresholds** (calibrated on training data):
  - **High Risk** ≥ 65% (3-5 machines per 500 typical)
  - **Medium Risk** 40-65% (400-470 machines per 500 typical)
  - **Low Risk** < 40% (30-100 machines per 500 typical)

### 2. **Explainability (Feature Attribution)**
- **Method**: Weight × Input interaction (first layer analysis)
- **Output**: Top 3 contributing factors with impact scores
- **Example Response**:
  ```json
  {
    "feature": "L0_S1_F6",
    "impact_score": 0.4521
  }
  ```
- **Why this approach**:
  - Works with PyTorch models
  - Computationally efficient (no sampling)
  - Interpretable + interview-ready
  - Foundation for SHAP enhancement

### 3. **Intelligent Action Recommendations**
Rules-based mapping of risk to domain-specific maintenance:

| Priority | Base Actions | Feature-Specific Rules |
|----------|-------------|----------------------|
| **High** | Schedule immediate inspection (24h) | Inspect bearings (vibration) |
| | Reduce load to 70% | Check cooling (temperature) |
| | Prepare spare parts | Verify valves (pressure) |
| **Medium** | Schedule next maintenance window | + feature-specific tasks |
| **Low** | Continue monitoring | Routine maintenance |

### 4. **Delay Impact Quantification**
Contextual messaging on consequences:
```
HIGH RISK:
- Cascading failures (cost multiplier: 100-200%)
- Production downtime ($K+/hour)
- Safety hazards
→ Action urgently required
```

### 5. **What-If Simulation**
Endpoint: `POST /simulate`
```json
{
  "base_features": { ...current data... },
  "modifications": { "vibration": 0.2, "temperature": 25 }
}
```
**Response**: New risk score + delta + improvement indicator

**Use case**: "If we reduce vibration from 0.8 to 0.2, how much will risk drop?"

---

## 🚀 How to Run

### **Backend Setup**

```bash
# Activate venv
cd "C:\Users\meghn\Downloads\SMART MANUFACTURING GENAI"
.\.venv\Scripts\Activate.ps1

# Start Flask
cd Backend\app\api
python main.py
# Server runs on http://127.0.0.1:5000
```

### **Frontend Setup**

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### **Test Endpoints**

**1. Health Check**
```bash
curl http://127.0.0.1:5000/health
```

**2. Single Prediction**
```bash
curl -X POST http://127.0.0.1:5000/predict/torch \
  -H "Content-Type: application/json" \
  -d '{"L0_S1_F0": -1.61, "L0_S1_F1": 0.32, ...}'
```

**3. Batch CSV Upload**
- Use frontend UI or:
```bash
curl -X POST http://127.0.0.1:5000/upload-csv \
  -F "file=@data.csv"
```

**4. What-If Simulation**
```bash
curl -X POST http://127.0.0.1:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "base_features": {...},
    "modifications": {"vibration": 0.2}
  }'
```

---

## 📈 Typical Output (CSV Upload)

Each row returns:
```json
{
  "row_index": 0,
  "failure_probability": 0.5398,
  "risk_score": 53,
  "maintenance_priority": "Medium",
  "urgency": "WARNING",
  "timeline": "Within 7 days",
  "top_risk_factors": [
    {
      "feature": "L0_S1_F6",
      "impact_score": 0.4521
    },
    {
      "feature": "L0_S1_F8",
      "impact_score": 0.3892
    },
    {
      "feature": "cat_var_0",
      "impact_score": 0.2145
    }
  ],
  "recommended_actions": [
    "🔧 Inspect bearings and alignment",
    "Schedule inspection during next maintenance window (1-7 days)",
    "Increase monitoring frequency"
  ]
}
```

---

## 🎓 Why This Design?

### **Percentile-Based Risk (Not Fixed Thresholds)**
- Real data: 54.8% avg failure probability across fleet
- **Fixed 75% threshold would find ZERO high-risk machines** ❌
- **Calibrated 65% threshold identifies ~3 critical machines** ✅
- Reflects operational reality, not arbitrary numbers

### **Rule-Based Actions (Not ML-Only)**
- ML predicts risk
- Domain knowledge (rules) recommends actions
- Combined approach mirrors real maintenance decision-making
- Auditable + explainable

### **Modular Architecture**
- `explainability.py` — Separates attribution logic
- `action_engine.py` — Isolates business rules
- Easy to update without touching core API
- Extensible for SHAP integration

---

## 🔮 Advanced Features (Optional)

### **SHAP Integration**
Replace weight-based explanation with SHAP:
```python
import shap
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(x)
```

### **Percentile Calibration**
Instead of fixed 65/40 thresholds, use historical percentiles:
```python
percentiles = {
    'p75': 0.67,   # 75th percentile from training
    'p40': 0.39    # 40th percentile from training
}
```

### **Anomaly Detection Layer**
Flag machines with unusual feature combinations (one-class SVM, Isolation Forest).

---

## 📊 Project Structure

```
Backend/
  ├── app/
  │   ├── api/
  │   │   └── main.py              (Flask app + endpoints)
  │   ├── ml/
  │   │   ├── explainability.py    (Feature attribution)
  │   │   ├── action_engine.py     (Recommendation rules)
  │   │   ├── preprocessing.py     (Data pipelines)
  │   │   ├── train_torch.py       (Model training)
  │   │   └── ...
  │   └── models_artifacts/
  │       ├── torch_failure_model_best.pt
  │       ├── torch_scaler.pkl
  │       ├── torch_label_encoders.pkl
  │       └── torch_columns.pkl
  └── data/
      ├── merged_train_reduced.csv
      ├── merged_test_reduced.csv
      └── sample_submission_reduced.csv

frontend/
  ├── src/
  │   ├── App.jsx
  │   ├── components/
  │   │   ├── ResultsTable.jsx     (Expandable rows)
  │   │   ├── SummaryPanel.jsx
  │   │   └── CsvUpload.jsx
  │   └── services/
  │       └── api.js
  └── package.json
```

---

## 🛠️ Environment Variables

**None required for MVP**, but for production consider:
```env
FLASK_ENV=production
MODEL_PATH=/path/to/models
LOG_LEVEL=INFO
SIMULATION_ENABLED=true
```

---

## ✅ Success Criteria Met

- ✅ **Standardized Output Schema** — All predictions follow consistent structure
- ✅ **Feature Attribution** — Top 3 contributing factors with scores
- ✅ **Maintenance Recommendations** — Smart, rule-based actions
- ✅ **Explainability UI** — Expandable rows with full context
- ✅ **What-If Simulation** — /simulate endpoint for proactive decisions
- ✅ **Production-Ready** — Error handling, logging, modular code
- ✅ **Interview-Ready** — Clear architecture, defensible choices

---

## 📝 Future Enhancements

1. **SHAP-Based Explanations** — Model-agnostic feature importance
2. **A/B Testing Framework** — Validate rule effectiveness
3. **Feedback Loop** — Train on maintenance outcomes
4. **Real-Time Streaming** — Kafka integration for continuous data
5. **Anomaly Detection** — Flag unusual machines before risks spike
6. **Multi-Model Ensemble** — XGBoost + Random Forest baselines

---

## 📞 Support

For questions on the architecture or implementation, see:
- **Backend Logic**: `Backend/app/api/main.py`
- **Explainability**: `Backend/app/ml/explainability.py`
- **Actions**: `Backend/app/ml/action_engine.py`
- **Frontend**: `frontend/src/App.jsx`

---

**Built with PyTorch + Flask + React + TailwindCSS**  
*A complete predictive maintenance decision engine, not just a model.*
