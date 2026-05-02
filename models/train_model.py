"""
Nagar Alert Hub - ML Model Training Pipeline
Trains Logistic Regression classifier for incident risk classification.
Run this once to generate the model artifacts.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── Synthetic Training Data ──────────────────────────────────────────
HIGH_RISK = [
    "bomb explosion near market area massive casualties",
    "murder committed in residential colony victim found dead",
    "acid attack on woman near bus stand critical condition",
    "armed robbery at bank gunshots fired hostages taken",
    "terrorist attack on public gathering multiple killed",
    "fire engulfing building people trapped inside",
    "gang war shootout in downtown area several injured",
    "sexual assault reported victim in hospital",
    "kidnapping of minor child amber alert issued",
    "gas leak explosion in industrial area workers critical",
    "stabbing incident at railway station victim bleeding",
    "mob violence riot breaking out police deployed",
    "suicide attempt from building roof emergency services needed",
    "drug trafficking bust armed smugglers dangerous cargo",
    "child abuse case serious injuries reported to police",
    "fatal road accident multiple deaths on highway",
    "drowning incident river flood victims missing",
    "building collapse construction site workers buried",
    "chemical spill toxic fumes spreading evacuate area",
    "sniper attack on police officers officer down",
    "human trafficking ring busted victims rescued",
    "arson fire deliberately set to residential complex",
    "hate crime attack racially motivated assault hospitalised",
    "domestic violence severe beating wife hospitalised critical",
    "gunman spotted near school children in danger lockdown",
    "poisoning incident food contamination mass hospitalisation",
    "extortion threat with weapon business owner attacked",
    "road rage murder driver killed on highway",
    "grenade thrown at police station major blast",
    "abduction attempt on child near school foiled",
]

LOW_RISK = [
    "stray dog menace near park residents complaining",
    "garbage not collected for three days bad smell",
    "streetlight not working on main road since week",
    "pothole on road causing minor accidents near signal",
    "water supply disrupted in colony pipes broken",
    "noise complaint about loud music from neighbour",
    "illegal parking blocking entrance to hospital",
    "tree fallen on road causing traffic disruption",
    "minor scuffle between neighbours over parking space",
    "power outage in area for several hours",
    "construction noise disturbing residents at night",
    "stray cattle blocking traffic on highway",
    "drainage overflow dirty water on road",
    "vandalism of public property graffiti on walls",
    "chain snatching attempt failed no injury",
    "eve teasing complaint registered at police station",
    "traffic signal not working vehicles causing jam",
    "public toilet not clean complaint filed online",
    "overgrown trees blocking street visibility",
    "minor accident between two bikes no injuries",
    "pickpocketing at market area wallet stolen",
    "suspicious person loitering near ATM reported",
    "broken footpath causing difficulty for pedestrians",
    "water logging after rain in low lying area",
    "dog bite minor wound first aid given",
    "shopkeeper dispute over prices argument recorded",
    "children playing cricket damaging vehicle complaint",
    "illegal vendor blocking footpath removal requested",
    "mobile phone snatching victim unharmed item recovered",
    "building maintenance issue leak in roof reported",
]

# Augment dataset
def augment(samples, n=3):
    augmented = []
    for s in samples:
        words = s.split()
        for _ in range(n):
            np.random.shuffle(words)
            augmented.append(" ".join(words))
    return augmented

high_aug = augment(HIGH_RISK, 4)
low_aug = augment(LOW_RISK, 4)

texts = HIGH_RISK + high_aug + LOW_RISK + low_aug
labels = (
    ['high_risk'] * (len(HIGH_RISK) + len(high_aug)) +
    ['low_risk'] * (len(LOW_RISK) + len(low_aug))
)

df = pd.DataFrame({'text': texts, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['text_clean'] = df['text'].apply(preprocess)

X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── MLflow Experiment ────────────────────────────────────────────────
os.makedirs("mlflow_runs", exist_ok=True)
mlflow.set_tracking_uri("file:./mlflow_runs")
mlflow.set_experiment("nagar_alert_hub")

models = {
    "LogisticRegression": LogisticRegression(max_iter=500, C=1.0, random_state=42),
    "NaiveBayes": MultinomialNB(alpha=0.5),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LinearSVC": LinearSVC(max_iter=1000, random_state=42),
}

best_acc = 0
best_pipeline = None

for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True
            )),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        mlflow.sklearn.log_model(pipeline, name)

        print(f"{name}: acc={acc:.3f} | cv={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

        if acc > best_acc:
            best_acc = acc
            best_pipeline = pipeline
            best_name = name

print(f"\n✅ Best model: {best_name} (accuracy={best_acc:.3f})")

# ── Save artefacts ───────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/incident_classifier.pkl")

meta = {
    "model_name": best_name,
    "accuracy": round(best_acc, 4),
    "classes": list(best_pipeline.classes_),
    "version": "1.0.0",
    "features": 5000,
}
with open("models/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Model saved to models/incident_classifier.pkl")
print("✅ Metadata saved to models/model_meta.json")
