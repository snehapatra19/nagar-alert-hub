"""
Nagar Alert Hub - ML Model Training Pipeline
Real datasets: 95k + 11k + 7k + 8k = ~122k rows
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW = True
except Exception:
    MLFLOW = False

BASE = os.path.dirname(os.path.abspath(__file__))

HIGH_RISK_CATEGORIES = {
    'Earthquake','Flood','Hurricane','Cyclone','Tsunami',
    'Typhoon','Volcanic Eruption','Wildfire','Industrial Accident','Drought'
}

def clean(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

all_texts, all_labels = [], []

print("Loading datasets...")

# 1. final_dataset_mini_balanced.csv
try:
    df1 = pd.read_csv(os.path.join(BASE, 'final_dataset_mini_balanced.csv'))
    df1['tc'] = df1['text'].apply(clean)
    df1['risk'] = df1['label'].apply(lambda x: 'high_risk' if x in HIGH_RISK_CATEGORIES else 'low_risk')
    all_texts.extend(df1['tc'].tolist())
    all_labels.extend(df1['risk'].tolist())
    print(f"  final_dataset: {len(df1)} rows")
except Exception as e:
    print(f"  final_dataset error: {e}")

# 2. disaster_tweets.csv
try:
    df2 = pd.read_csv(os.path.join(BASE, 'disaster_tweets.csv'), encoding='utf-8-sig')
    df2['tc'] = df2['text'].apply(clean)
    df2['risk'] = df2['target'].apply(lambda x: 'high_risk' if str(x)=='1' else 'low_risk')
    all_texts.extend(df2['tc'].tolist())
    all_labels.extend(df2['risk'].tolist())
    print(f"  disaster_tweets: {len(df2)} rows")
except Exception as e:
    print(f"  disaster_tweets error: {e}")

# 3. crime_dataset.csv
try:
    df3 = pd.read_csv(os.path.join(BASE, 'crime_dataset.csv'))
    df3['tc'] = df3['title'].apply(clean)
    df3['risk'] = df3['is_crime_report'].apply(lambda x: 'high_risk' if str(x)=='1' else 'low_risk')
    all_texts.extend(df3['tc'].tolist())
    all_labels.extend(df3['risk'].tolist())
    print(f"  crime_dataset: {len(df3)} rows")
except Exception as e:
    print(f"  crime_dataset error: {e}")

# 4. old_train_dataset.csv
try:
    df4 = pd.read_csv(os.path.join(BASE, 'old_train_dataset.csv'))
    df4['tc'] = df4['text'].apply(clean)
    df4['risk'] = df4['target'].apply(lambda x: 'high_risk' if str(x)=='1' else 'low_risk')
    all_texts.extend(df4['tc'].tolist())
    all_labels.extend(df4['risk'].tolist())
    print(f"  old_train_dataset: {len(df4)} rows")
except Exception as e:
    print(f"  old_train_dataset error: {e}")

# Seed data x15
HIGH_SEED = [
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
    "chemical spill toxic fumes spreading evacuate area",
    "grenade thrown at police station major blast",
    "gunman spotted near school children in danger lockdown",
    "human trafficking ring busted victims rescued",
    "arson fire deliberately set to residential complex",
    "domestic violence severe beating wife hospitalised critical",
    "building collapse construction site workers buried",
    "drowning incident river flood victims missing",
    "fatal road accident multiple deaths on highway",
]
LOW_SEED = [
    "stray dog menace near park residents complaining",
    "garbage not collected for three days bad smell",
    "streetlight not working on main road since week",
    "pothole on road causing minor accidents near signal",
    "water supply disrupted in colony pipes broken",
    "noise complaint about loud music from neighbour",
    "illegal parking blocking entrance to hospital",
    "tree fallen on road causing traffic disruption",
    "power outage in area for several hours",
    "construction noise disturbing residents at night",
    "drainage overflow dirty water on road",
    "traffic signal not working vehicles causing jam",
    "minor accident between two bikes no injuries",
    "pickpocketing at market area wallet stolen",
    "stray cattle blocking traffic on highway",
    "broken footpath causing difficulty for pedestrians",
    "water logging after rain in low lying area",
    "mobile phone snatching victim unharmed item recovered",
    "public toilet not clean complaint filed online",
    "overgrown trees blocking street visibility",
]

for _ in range(15):
    all_texts.extend([clean(t) for t in HIGH_SEED])
    all_labels.extend(['high_risk']*len(HIGH_SEED))
    all_texts.extend([clean(t) for t in LOW_SEED])
    all_labels.extend(['low_risk']*len(LOW_SEED))

df = pd.DataFrame({'text': all_texts, 'label': all_labels})
df = df[df['text'].str.len() > 5].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTotal dataset: {len(df):,} rows")
print(f"High risk: {(df['label']=='high_risk').sum():,}")
print(f"Low risk:  {(df['label']=='low_risk').sum():,}")

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

if MLFLOW:
    os.makedirs(os.path.join(BASE,'..','mlflow_runs'), exist_ok=True)
    mlflow.set_tracking_uri(f"file:{os.path.join(BASE,'..','mlflow_runs')}")
    mlflow.set_experiment("nagar_alert_hub_realdata")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=2.0, random_state=42),
    "NaiveBayes": MultinomialNB(alpha=0.3),
    "LinearSVC": LinearSVC(max_iter=2000, random_state=42),
}

best_acc, best_pipeline, best_name = 0, None, None

print("\nTraining models...")
for name, clf in models.items():
    try:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000, ngram_range=(1,2),
                min_df=2, sublinear_tf=True, strip_accents='unicode'
            )),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  {name}: accuracy={acc:.4f}")
        print(classification_report(y_test, y_pred))
        if MLFLOW:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(pipeline, name)
        if acc > best_acc:
            best_acc, best_pipeline, best_name = acc, pipeline, name
    except Exception as e:
        print(f"  {name} failed: {e}")

print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")

joblib.dump(best_pipeline, os.path.join(BASE, 'incident_classifier.pkl'))

meta = {
    "model_name": best_name,
    "accuracy": round(best_acc, 4),
    "classes": list(best_pipeline.classes_),
    "version": "2.0.0",
    "features": 10000,
    "training_samples": len(X_train),
    "datasets": [
        "final_dataset_mini_balanced.csv (95k)",
        "disaster_tweets.csv (11k)",
        "crime_dataset.csv (7k)",
        "old_train_dataset.csv (8k)"
    ]
}
with open(os.path.join(BASE, 'model_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"Model saved! Trained on {len(df):,} real incidents!")