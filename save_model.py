"""
save_model.py — Run this ONCE before deploying to Streamlit Cloud.
It trains the Random Forest on your F1 CSVs and saves rf_model.pkl + model_meta.pkl
so the app loads instantly without retraining on every visit.

Usage:
    python save_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import time

print("=" * 55)
print("  F1 Model Trainer — save_model.py")
print("=" * 55)

# ── Load & merge ──────────────────────────────────────────────────
print("\n[1/4] Loading CSVs ...")
results      = pd.read_csv('results.csv')
races        = pd.read_csv('races.csv')[['raceId','year','circuitId']]
constructors = pd.read_csv('constructors.csv')[['constructorId','name']].rename(columns={'name':'constructor_name'})
drivers      = pd.read_csv('drivers.csv')[['driverId','driverRef']]
circuits     = pd.read_csv('circuits.csv')[['circuitId','name']].rename(columns={'name':'circuit_name'})

df = (results[['raceId','driverId','constructorId','grid','laps','positionOrder','points']]
      .merge(races, on='raceId')
      .merge(constructors, on='constructorId')
      .merge(drivers, on='driverId')
      .merge(circuits, on='circuitId'))

df.replace('\\N', np.nan, inplace=True)
for col in ['grid','laps','positionOrder','points']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=['positionOrder','grid','laps'], inplace=True)
df['podium'] = (df['positionOrder'] <= 3).astype(int)
print(f"   Rows: {len(df):,}  |  Podium rate: {df['podium'].mean()*100:.1f}%")

# ── Encode ────────────────────────────────────────────────────────
print("\n[2/4] Encoding features ...")

le_con = LabelEncoder(); df['constructorId_enc'] = le_con.fit_transform(df['constructorId'].astype(str))
le_drv = LabelEncoder(); df['driverId_enc']      = le_drv.fit_transform(df['driverId'].astype(str))
le_cir = LabelEncoder(); df['circuitId_enc']     = le_cir.fit_transform(df['circuitId'].astype(str))

df['front_row']  = (df['grid'] <= 3).astype(int)
max_laps         = df.groupby('raceId')['laps'].transform('max')
df['laps_ratio'] = (df['laps'] / max_laps.replace(0, np.nan)).fillna(0)

constructor_list = sorted(df['constructor_name'].dropna().unique().tolist())
driver_list      = sorted(df['driverRef'].dropna().unique().tolist())
circuit_list     = sorted(df['circuit_name'].dropna().unique().tolist())

FEATURES = ['grid','year','front_row','laps_ratio',
            'driverId_enc','constructorId_enc','circuitId_enc']
X = df[FEATURES]
y = df['podium']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Train ─────────────────────────────────────────────────────────
print("\n[3/4] Training Random Forest ...")
t0 = time.perf_counter()
model = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    max_features='sqrt', class_weight='balanced',
    random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
t = time.perf_counter() - t0
print(f"   Training time: {t:.2f}s")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print(f"   Test F1-macro : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"   Test ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

# ── Save ──────────────────────────────────────────────────────────
print("\n[4/4] Saving model & metadata ...")
meta = {
    'constructors'    : constructor_list,
    'drivers'         : driver_list,
    'circuits'        : circuit_list,
    'constructor_map' : dict(zip(df['constructor_name'].astype(str), df['constructorId_enc'].astype(int))),
    'driver_map'      : dict(zip(df['driverRef'].astype(str), df['driverId_enc'].astype(int))),
    'circuit_map'     : dict(zip(df['circuit_name'].astype(str), df['circuitId_enc'].astype(int))),
    'feature_names'   : FEATURES,
    'year_min'        : int(df['year'].min()),
    'year_max'        : int(df['year'].max()),
}

joblib.dump(model, 'rf_model.pkl')
joblib.dump(meta,  'model_meta.pkl')
print("   Saved: rf_model.pkl")
print("   Saved: model_meta.pkl")
print("\n✅ Done! Now push rf_model.pkl + model_meta.pkl to GitHub with your app.")
