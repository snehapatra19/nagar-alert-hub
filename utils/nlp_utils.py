"""
NLP preprocessing utilities for Nagar Alert Hub
"""

import re
import json

# ── Keyword Override Rules ───────────────────────────────────────────
CRITICAL_KEYWORDS = {
    "bomb": 0.99,
    "explosion": 0.97,
    "blast": 0.95,
    "murder": 0.99,
    "killed": 0.95,
    "acid attack": 0.99,
    "terrorist": 0.99,
    "terrorism": 0.98,
    "hostage": 0.97,
    "gunshot": 0.97,
    "shooting": 0.96,
    "stabbing": 0.95,
    "rape": 0.99,
    "sexual assault": 0.99,
    "kidnap": 0.98,
    "abduction": 0.98,
    "grenade": 0.99,
    "arson": 0.95,
    "fire engulf": 0.96,
    "building collapse": 0.96,
    "chemical spill": 0.94,
    "toxic gas": 0.95,
    "child abuse": 0.98,
    "trafficking": 0.97,
    "sniper": 0.99,
    "suicide attempt": 0.94,
    "drowning": 0.92,
    "fatal": 0.93,
    "dead body": 0.97,
    "unconscious": 0.88,
    "critical condition": 0.90,
}

HIGH_RISK_ACTIONS = {
    "bomb": ["Evacuate area immediately", "Alert NSG/ATS", "Cordon 500m radius"],
    "murder": ["Secure crime scene", "Alert homicide unit", "Medical examiner"],
    "acid attack": ["Rush to burn unit", "Alert women cell", "FIR registration"],
    "fire": ["Alert fire brigade 101", "Evacuate building", "Check for injuries"],
    "kidnap": ["Activate AMBER alert", "Alert CBI missing unit", "CCTV review"],
    "shooting": ["Deploy armed response", "Evacuate civilians", "Alert trauma center"],
    "default": ["Deploy rapid response team", "Alert district control room", "Establish perimeter"],
}

AUTHORITIES = {
    "high_risk": [
        {"name": "Police Control Room", "number": "100", "icon": "🚔"},
        {"name": "Fire Brigade", "number": "101", "icon": "🚒"},
        {"name": "Ambulance", "number": "108", "icon": "🚑"},
        {"name": "Disaster Management", "number": "1077", "icon": "🆘"},
        {"name": "Women Helpline", "number": "1091", "icon": "👮"},
    ],
    "low_risk": [
        {"name": "Local Police Station", "number": "100", "icon": "🚔"},
        {"name": "Municipal Helpline", "number": "1533", "icon": "🏛️"},
        {"name": "Traffic Police", "number": "103", "icon": "🚦"},
    ],
}


def preprocess_text(text: str) -> str:
    """Clean and normalize incident text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def keyword_override(text: str):
    """
    Check for critical keywords. Returns (confidence, matched_keyword) or None.
    """
    text_lower = text.lower()
    best_conf = 0
    best_kw = None
    for kw, conf in CRITICAL_KEYWORDS.items():
        if kw in text_lower and conf > best_conf:
            best_conf = conf
            best_kw = kw
    if best_conf > 0:
        return best_conf, best_kw
    return None


def get_recommended_actions(label: str, text: str, keyword: str = None) -> list:
    """Return context-aware recommended actions."""
    if label == "low_risk":
        return [
            "File online complaint at local police portal",
            "Contact municipal corporation helpline",
            "Document evidence with photos/videos",
            "Follow up with ward officer if unresolved",
        ]
    # High risk
    if keyword:
        for k, actions in HIGH_RISK_ACTIONS.items():
            if k in (keyword or ""):
                return actions
    return HIGH_RISK_ACTIONS["default"]


def get_authorities(label: str) -> list:
    return AUTHORITIES.get(label, AUTHORITIES["low_risk"])


def get_explanation(label: str, confidence: float, keyword: str = None) -> str:
    if keyword:
        return (
            f"⚠️ Critical keyword detected: '{keyword}'. "
            f"System flagged as HIGH RISK with {confidence*100:.1f}% confidence. "
            "Immediate response protocol activated."
        )
    if label == "high_risk":
        return (
            f"ML model classified this incident as HIGH RISK ({confidence*100:.1f}% confidence). "
            "The text pattern matches severe threat categories in the training dataset."
        )
    return (
        f"ML model classified this as LOW RISK ({confidence*100:.1f}% confidence). "
        "Standard municipal response recommended. Monitor for escalation."
    )
