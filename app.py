"""
Nagar Alert Hub — Flask Backend
Production-ready public safety incident classification system.
"""

import os
import re
import json
import logging
import joblib
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── App Setup ────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

BASE_DIR = Path(__file__).parent
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nagar-alert-hub-secret-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL',
    f"sqlite:///{BASE_DIR / 'instance' / 'incidents.db'}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ── Load ML Model ────────────────────────────────────────────────────
MODEL_PATH = BASE_DIR / 'models' / 'incident_classifier.pkl'
META_PATH = BASE_DIR / 'models' / 'model_meta.json'

pipeline = None
model_meta = {}

def load_model():
    global pipeline, model_meta
    try:
        pipeline = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            with open(META_PATH) as f:
                model_meta = json.load(f)
        logger.info(f"✅ Model loaded: {model_meta.get('model_name', 'unknown')}")
    except FileNotFoundError:
        logger.warning("⚠️ Model not found. Run models/train_model.py first.")
        pipeline = None

load_model()

import sys
sys.path.insert(0, str(BASE_DIR))
from utils.nlp_utils import (
    preprocess_text, keyword_override,
    get_recommended_actions, get_authorities, get_explanation
)

# ── Database Models ──────────────────────────────────────────────────
class Incident(db.Model):
    __tablename__ = 'incidents'

    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.Text, nullable=False)
    location_name = db.Column(db.String(200))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    risk_label = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    keyword_triggered = db.Column(db.String(100))
    reporter_name = db.Column(db.String(100), default='Anonymous')
    status = db.Column(db.String(20), default='open')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    image_path = db.Column(db.String(300))

    def to_dict(self):
        return {
            'id': self.id,
            'description': self.description,
            'location_name': self.location_name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'risk_label': self.risk_label,
            'confidence': round(self.confidence * 100, 1) if self.confidence else None,
            'keyword_triggered': self.keyword_triggered,
            'reporter_name': self.reporter_name,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


with app.app_context():
    os.makedirs(BASE_DIR / 'instance', exist_ok=True)
    db.create_all()

# ── Prediction Core ──────────────────────────────────────────────────
def classify_incident(text: str):
    override = keyword_override(text)

    if override:
        confidence, kw = override
        label = 'high_risk'
        source = 'keyword_override'
    elif pipeline is not None:
        clean = preprocess_text(text)
        try:
            # LogisticRegression / NaiveBayes support predict_proba
            proba = pipeline.predict_proba([clean])[0]
            classes = pipeline.classes_
            idx = np.argmax(proba)
            label = classes[idx]
            confidence = float(proba[idx])
        except AttributeError:
            # LinearSVC does not support predict_proba
            label = pipeline.predict([clean])[0]
            decision = pipeline.decision_function([clean])[0]
            confidence = float(1 / (1 + np.exp(-abs(decision))))
        kw = None
        source = 'ml_model'
    else:
        label = 'low_risk'
        confidence = 0.6
        kw = None
        source = 'fallback'

    actions = get_recommended_actions(label, text, kw)
    authorities = get_authorities(label)
    explanation = get_explanation(label, confidence, kw)

    return {
        'label': label,
        'confidence': confidence,
        'keyword': kw,
        'source': source,
        'actions': actions,
        'authorities': authorities,
        'explanation': explanation,
    }

# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           google_maps_key=os.getenv('GOOGLE_MAPS_API_KEY', ''))


@app.route('/dashboard')
def dashboard():
    incidents = Incident.query.order_by(Incident.created_at.desc()).limit(50).all()
    stats = {
        'total': Incident.query.count(),
        'high_risk': Incident.query.filter_by(risk_label='high_risk').count(),
        'low_risk': Incident.query.filter_by(risk_label='low_risk').count(),
        'open': Incident.query.filter_by(status='open').count(),
        'resolved': Incident.query.filter_by(status='resolved').count(),
    }
    return render_template('dashboard.html',
                           incidents=incidents,
                           stats=stats,
                           google_maps_key=os.getenv('GOOGLE_MAPS_API_KEY', ''))


@app.route('/report')
def report_page():
    return render_template('index.html',
                           google_maps_key=os.getenv('GOOGLE_MAPS_API_KEY', ''))


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ── API Endpoints ─────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '').strip()

        if not text or len(text) < 5:
            return jsonify({'error': 'Incident description too short'}), 400

        result = classify_incident(text)

        incident = Incident(
            description=text,
            location_name=data.get('location_name'),
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            reporter_name=data.get('reporter_name', 'Anonymous'),
            risk_label=result['label'],
            confidence=result['confidence'],
            keyword_triggered=result['keyword'],
        )
        db.session.add(incident)
        db.session.commit()

        logger.info(f"Incident #{incident.id} | {result['label']} | {result['confidence']:.2f}")

        return jsonify({
            'incident_id': incident.id,
            **result,
            'confidence_pct': round(result['confidence'] * 100, 1),
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/incidents', methods=['GET'])
def get_incidents():
    limit = min(int(request.args.get('limit', 100)), 500)
    label = request.args.get('label')
    status = request.args.get('status')

    q = Incident.query
    if label:
        q = q.filter_by(risk_label=label)
    if status:
        q = q.filter_by(status=status)

    incidents = q.order_by(Incident.created_at.desc()).limit(limit).all()
    return jsonify([i.to_dict() for i in incidents])


@app.route('/api/incidents/<int:incident_id>', methods=['GET'])
def get_incident(incident_id):
    inc = Incident.query.get_or_404(incident_id)
    return jsonify(inc.to_dict())


@app.route('/api/incidents/<int:incident_id>/status', methods=['PATCH'])
def update_status(incident_id):
    inc = Incident.query.get_or_404(incident_id)
    data = request.get_json()
    new_status = data.get('status')
    if new_status not in ('open', 'reviewing', 'resolved'):
        return jsonify({'error': 'Invalid status'}), 400
    inc.status = new_status
    db.session.commit()
    return jsonify({'success': True, 'status': new_status})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    total = Incident.query.count()
    high = Incident.query.filter_by(risk_label='high_risk').count()
    low = Incident.query.filter_by(risk_label='low_risk').count()
    open_count = Incident.query.filter_by(status='open').count()
    resolved = Incident.query.filter_by(status='resolved').count()

    from sqlalchemy import func
    trend = db.session.query(
        func.date(Incident.created_at).label('date'),
        func.count(Incident.id).label('count')
    ).group_by(func.date(Incident.created_at)).order_by('date').limit(7).all()

    return jsonify({
        'total': total,
        'high_risk': high,
        'low_risk': low,
        'open': open_count,
        'resolved': resolved,
        'model': model_meta,
        'trend': [{'date': str(t.date), 'count': t.count} for t in trend],
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify(model_meta)


@app.route('/result/<int:incident_id>')
def result_page(incident_id):
    inc = Incident.query.get_or_404(incident_id)
    result = classify_incident(inc.description)
    return render_template('result.html', incident=inc, result=result,
                           google_maps_key=os.getenv('GOOGLE_MAPS_API_KEY', ''))


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)