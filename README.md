# 🛡️ Nagar Alert Hub
### AI-Powered Public Safety Monitoring System

> Real-time incident reporting, NLP classification, and live dashboard for smart-city emergency response.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-blue.svg)](https://mlflow.org)

---

## ✨ Features

- **AI Classification** — Logistic Regression + TF-IDF vectorizer (85-90% accuracy)
- **Keyword Override** — 30+ critical terms (bomb, murder, acid attack) instant HIGH RISK
- **Google Maps + Leaflet** — Live incident map with geolocation
- **Real-time Dashboard** — Auto-refreshing command center with charts
- **MLflow Integration** — Model versioning, experiment tracking
- **REST API** — Full JSON API for integration
- **Deployment Ready** — Render, Railway, Heroku, Docker, VPS

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone or unzip the project
cd nagar-alert-hub

# 2. One-command setup and launch
chmod +x start.sh && ./start.sh

# 3. Open browser
open http://localhost:5000
```

### Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the ML model
python models/train_model.py

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_MAPS_API_KEY

# Run development server
python app.py
```

---

## 🗺️ Google Maps Setup

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → Enable APIs:
   - **Maps JavaScript API**
   - **Geocoding API** (optional)
3. Create API Key → Restrict to your domain
4. Add to `.env`:
   ```
   GOOGLE_MAPS_API_KEY=AIza...your_key_here
   ```
5. Restart the server — Google Maps will be used automatically!

> Without a key, the app uses **Leaflet + OpenStreetMap** (free, no key needed).

---

## ☁️ Deployment Options

### Option 1: Render.com (Recommended — Free Tier)

1. Push code to GitHub:
   ```bash
   git init && git add . && git commit -m "Initial commit"
   gh repo create nagar-alert-hub --public --push
   ```
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` ✅
5. Add environment variable: `GOOGLE_MAPS_API_KEY`
6. Deploy! 🎉

**Your URL:** `https://nagar-alert-hub.onrender.com`

---

### Option 2: Railway.app (Easiest)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
railway open
```

Add env vars in Railway dashboard.

---

### Option 3: Heroku

```bash
# Install Heroku CLI
heroku create nagar-alert-hub
heroku config:set SECRET_KEY=your-secret
heroku config:set GOOGLE_MAPS_API_KEY=your-key
git push heroku main
heroku open
```

---

### Option 4: Docker

```bash
# Build
docker build -t nagar-alert-hub .

# Run
docker run -p 5000:5000 \
  -e SECRET_KEY=your-secret \
  -e GOOGLE_MAPS_API_KEY=your-key \
  nagar-alert-hub

# Or with docker-compose
docker-compose up --build
```

---

### Option 5: VPS / Ubuntu Server

```bash
# On your server
sudo apt update && sudo apt install python3-pip nginx -y

# Clone and setup
git clone https://github.com/yourname/nagar-alert-hub
cd nagar-alert-hub
pip3 install -r requirements.txt
python3 models/train_model.py

# Run with gunicorn
gunicorn app:app --workers=2 --bind=0.0.0.0:5000 --daemon

# Nginx config (optional, for custom domain)
sudo nano /etc/nginx/sites-available/nagar
# Add proxy_pass http://127.0.0.1:5000;
sudo nginx -t && sudo systemctl restart nginx
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Classify incident text |
| `/api/incidents` | GET | List all incidents |
| `/api/incidents/:id` | GET | Get incident detail |
| `/api/incidents/:id/status` | PATCH | Update status |
| `/api/stats` | GET | Dashboard statistics |
| `/api/model-info` | GET | ML model metadata |

### Example: Submit Incident
```bash
curl -X POST https://your-app.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bomb found near market, police called",
    "location_name": "MG Road, Bengaluru",
    "latitude": 12.9716,
    "longitude": 77.5946,
    "reporter_name": "Ravi Kumar"
  }'
```

Response:
```json
{
  "incident_id": 1,
  "label": "high_risk",
  "confidence": 0.99,
  "confidence_pct": 99.0,
  "keyword": "bomb",
  "source": "keyword_override",
  "actions": ["Evacuate area immediately", "Alert NSG/ATS", "Cordon 500m radius"],
  "authorities": [{"name": "Police Control Room", "number": "100", "icon": "🚔"}],
  "explanation": "⚠️ Critical keyword detected: 'bomb'..."
}
```

---

## 🧠 ML Pipeline

```
Input Text
   │
   ▼
Keyword Override Check (30 critical terms)
   │ (if no keyword)
   ▼
Text Preprocessing (lowercase, remove punctuation)
   │
   ▼
TF-IDF Vectorization (5000 features, bigrams)
   │
   ▼
Logistic Regression Classifier
   │
   ▼
Risk Label + Confidence Score
```

**Models Evaluated:**
| Model | Accuracy |
|-------|----------|
| Logistic Regression ✅ | ~88% |
| Naive Bayes | ~84% |
| Random Forest | ~86% |
| Linear SVC | ~87% |

---

## 📁 Project Structure

```
nagar-alert-hub/
├── app.py                    # Flask application
├── requirements.txt
├── Procfile                  # Heroku/Render
├── render.yaml               # Render deployment
├── railway.json              # Railway deployment
├── Dockerfile
├── start.sh                  # Local startup script
├── .env.example
├── models/
│   ├── train_model.py        # ML training pipeline
│   ├── incident_classifier.pkl
│   └── model_meta.json
├── utils/
│   └── nlp_utils.py          # NLP helpers
├── templates/
│   ├── base.html
│   ├── index.html            # Report form
│   ├── result.html           # Classification result
│   └── dashboard.html        # Command dashboard
├── static/
│   ├── css/main.css
│   └── js/main.js
├── instance/
│   └── incidents.db          # SQLite database
└── mlflow_runs/              # MLflow artifacts
```

---

## 🎓 College Presentation Tips

1. **Demo flow**: Submit a report → Show result page → Show dashboard map
2. **Highlight**: Try "bomb explosion at railway station" to trigger keyword override
3. **Show** the MLflow experiment logs: `mlflow ui` in project folder
4. **API demo**: Open `/api/incidents` in browser to show JSON output

---

## 📞 Emergency Numbers (India)

| Service | Number |
|---------|--------|
| Police | 100 |
| Fire | 101 |
| Ambulance | 108 |
| Disaster | 1077 |
| Women Helpline | 1091 |
| Child Helpline | 1098 |

---

*Built with ❤️ for Smart City Safety · Nagar Alert Hub v1.0*
